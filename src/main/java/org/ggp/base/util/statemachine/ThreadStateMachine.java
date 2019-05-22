package org.ggp.base.util.statemachine;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.lucene.util.OpenBitSet;
import org.ggp.base.player.gamer.statemachine.XNodeAbstract;
import org.ggp.base.util.gdl.grammar.Gdl;
import org.ggp.base.util.gdl.grammar.GdlSentence;
import org.ggp.base.util.propnet.architecture.XPropNet;
import org.ggp.base.util.statemachine.exceptions.GoalDefinitionException;
import org.ggp.base.util.statemachine.exceptions.MoveDefinitionException;
import org.ggp.base.util.statemachine.exceptions.TransitionDefinitionException;



@SuppressWarnings("unused")
public class ThreadStateMachine extends XMachine {
    private OpenBitSet currentState, nextState, currInputs, currLegals, initState;

    private IntQueue q;
    public int[] components;
    public long[] compInfo;
    public int[] connecTable;
    private int self_index;

    private XStateMachine machine;

    private XORShiftRandom rand;

    public ThreadStateMachine(XStateMachine x, int ind) {
    	this.machine = x;
    	this.currentState = (OpenBitSet) x.getCurrentState().clone();
    	this.initState = (OpenBitSet) x.getCurrentState().clone();
    	this.nextState = (OpenBitSet) x.getNextState().clone();
    	this.currInputs = (OpenBitSet) x.getCurrInputs().clone();
    	this.currLegals = (OpenBitSet) x.getCurrLegals().clone();
    	this.components = Arrays.copyOf(x.getComponents(),x.getComponents().length);
    	this.compInfo = Arrays.copyOf(x.getCompInfo(), x.getCompInfo().length);
    	this.connecTable = Arrays.copyOf(x.getConnecTable(),x.getConnecTable().length);
    	this.rand = new XORShiftRandom();
    	this.q = new IntQueue(compInfo.length);
    	this.self_index = ind;
    }

    private static final int NUM_TYPE_BITS = 8;
	private static final int NUM_INPUT_BITS = 16;
	private static final int NUM_OUTPUT_BITS = 16;
	private static final int NUM_OFFSET_BITS = 24;

	private static final int OUTPUT_SHIFT = NUM_OFFSET_BITS;
	private static final int INPUT_SHIFT = OUTPUT_SHIFT + NUM_OUTPUT_BITS;
	private static final int TYPE_SHIFT = INPUT_SHIFT + NUM_INPUT_BITS;

	private static final long TYPE_MASK = 0xFF_0000_0000_000000L;
	private static final long INPUTS_MASK = 0x00_FFFF_0000_000000L;
    private static final long OUTPUTS_MASK = 0x00_0000_FFFF_000000L;
    private static final long OFFSET_MASK = 0x00_0000_0000_FFFFFFL;

    @Override
    public OpenBitSet getInitialState() {
    	return this.initState;
    }


	protected int type(long comp) {
    	return (int) ((comp & TYPE_MASK) >> TYPE_SHIFT);
    }


	protected int numInputs(long comp) {
    	return (int) ((comp & INPUTS_MASK) >> INPUT_SHIFT);
    }

	protected int numOutputs(long comp) {//inline these functions
    	return (int) ((comp & OUTPUTS_MASK) >> OUTPUT_SHIFT);
    }

	protected int outputsOffset(long comp) {
    	return (int) (comp & OFFSET_MASK);
    }

    private static final int CURR_VAL_MASK = 0x8000_0000;
    private static final int NOT_CURR_VAL_MASK = 0x7FFF_FFFF;

	protected boolean get_current_value(int value) {
    	return (value & CURR_VAL_MASK) != 0;
    }

    private static final long TRIGGER_MASK = 0x80_0000_0000_000000L;
    private static final long TRANSITION_MASK = 0x40_0000_0000_000000L;

	protected boolean isTrigger(long comp) {
    	return (comp & TRIGGER_MASK) != 0;
    }

    //Must be called after isTrigger
	protected boolean isTransition(long comp) {
    	return (comp & TRANSITION_MASK) != 0;
    }

    private static final int getId(int value) {
    	return (NOT_CURR_VAL_MASK & value);
    }

	protected void propagate() {
		System.out.println("Beginning propogate");
    	while(!(q.num_queued == 0)) {
    		int value = q.remove();
    		int compId = (NOT_CURR_VAL_MASK & value);

    		boolean val = (value & CURR_VAL_MASK) != 0;

    		long comp = compInfo[compId];
    		if ((comp & TRIGGER_MASK) != 0) {
    			if ((comp & TRANSITION_MASK) != 0) {
    				int outputIndex = (int) (comp & OFFSET_MASK);
    				int baseIndex = connecTable[outputIndex];
    				if (val) nextState.fastSet(baseIndex);
    				else nextState.clear(baseIndex);
    				continue;
    			} else {
    				int legalIndex = compId - machine.legalOffset;
    				if (val) currLegals.fastSet(legalIndex);
    				else currLegals.clear(legalIndex);
    			}
    		}

        	int num_outputs = (int) ((comp & OUTPUTS_MASK) >> OUTPUT_SHIFT);
        	int outputsIndex = (int) (comp & OFFSET_MASK);

        	for (int i = 0; i < num_outputs; ++i) {
        		int outIndex = connecTable[outputsIndex + i];
        		int outValue = components[outIndex];
        		int lastPropagatedValue = (outValue & CURR_VAL_MASK);

        		if (val) components[outIndex] += 1;
        		else components[outIndex] -= 1;

        		int newVal = (components[outIndex] & CURR_VAL_MASK);

        		if (newVal != lastPropagatedValue) {
        			q.add(newVal | outIndex);
        		}
        	}
    	}
    	q.clear();
		System.out.println("Ending propogate");
    }


	@Override
	public List<List<Move>> getLegalJointMoves(OpenBitSet state) throws MoveDefinitionException {
    	setState(state, null);

    	int size = machine.roles.length - 1;
        List<List<Move>> jointMoves = new ArrayList<List<Move>>();

    	for (int i = 0; i < size; ++i) {
    		List<Move> moves = new ArrayList<Move>();
    		int roleIndex = machine.rolesIndexMap[i];
    		int nextRoleIndex = machine.rolesIndexMap[i + 1];

    		for (int j = roleIndex; j < nextRoleIndex; ++j) {
    			if (currLegals.fastGet(j)) {
    				moves.add(machine.legalArray[j]);
    			}
    		}
    		jointMoves.add(moves);
    	}

    	int start = machine.rolesIndexMap[size];
    	int end = machine.legalArray.length;
    	List<Move> moves = new ArrayList<Move>();
    	for(int i = start; i < end; ++i) {
    		if (currLegals.fastGet(i)) {
    			moves.add(machine.legalArray[i]);
    		}
    	}
    	jointMoves.add(moves);

        List<List<Move>> crossProduct = new ArrayList<List<Move>>();
        crossProductLegalMoves(jointMoves, crossProduct, new ArrayDeque<Move>());//

        return crossProduct;
    }

	public List<List<Move>> getLegalJointMoves(OpenBitSet state, int rIndex, Move m) throws MoveDefinitionException {
    	setState(state, null);

    	int size = machine.roles.length;
        List<List<Move>> jointMoves = new ArrayList<List<Move>>();

    	for (int i = 0; i < rIndex; ++i) {
    		List<Move> moves = new ArrayList<Move>();
    		int roleIndex = machine.rolesIndexMap[i];
    		int nextRoleIndex = machine.rolesIndexMap[i + 1];

    		for (int j = roleIndex; j < nextRoleIndex; ++j) {
    			if (currLegals.fastGet(j)) {
    				moves.add(machine.legalArray[j]);
    			}
    		}
    		jointMoves.add(moves);
    	}

    	List<Move> rMoves = new ArrayList<Move>();
    	rMoves.add(m);
    	jointMoves.add(rMoves);

    	for (int i = rIndex + 1; i < size; ++i) {
    		List<Move> moves = new ArrayList<Move>();
    		int roleIndex = machine.rolesIndexMap[i];
    		int nextRoleIndex = (i == (size - 1) ? machine.legalArray.length : machine.rolesIndexMap[i + 1]);

    		for (int j = roleIndex; j < nextRoleIndex; ++j) {
    			if (currLegals.fastGet(j)) {
    				moves.add(machine.legalArray[j]);
    			}
    		}
    		jointMoves.add(moves);
    	}


        List<List<Move>> crossProduct = new ArrayList<List<Move>>();
        crossProductLegalMoves(jointMoves, crossProduct, new ArrayDeque<Move>());//

        return crossProduct;
    }

    @Override
	public List<Move> getRandomJointMove(OpenBitSet state) throws MoveDefinitionException
    {
    	setState(state, null);

    	int size = machine.roles.length - 1;
    	List<Move> randomJointMove = new ArrayList<Move>();
    	List<Move> moves;

    	for (int i = 0; i < size; ++i) {
    		moves = new ArrayList<Move>();
    		int roleIndex = machine.rolesIndexMap[i];
    		int nextRoleIndex = machine.rolesIndexMap[i + 1];

    		for (int j = roleIndex; j < nextRoleIndex; ++j) {
    			if (currLegals.fastGet(j)) {
    				moves.add(machine.legalArray[j]);
    			}
    		}
    		randomJointMove.add(moves.get(rand.nextInt(moves.size())));
    	}

    	int start = machine.rolesIndexMap[size];
    	int end = machine.legalArray.length;
    	moves = new ArrayList<Move>();
    	for(int i = start; i < end; ++i) {
    		if (currLegals.fastGet(i)) {
    			moves.add(machine.legalArray[i]);
    		}
    	}
    	randomJointMove.add(moves.get(rand.nextInt(moves.size())));
        return randomJointMove;

    }

    public List<Move> getRandomJointMoveCurr(OpenBitSet state) throws MoveDefinitionException
    {
    	//setState(state, null);

    	int size = machine.roles.length - 1;
    	List<Move> randomJointMove = new ArrayList<Move>();
    	List<Move> moves;

    	for (int i = 0; i < size; ++i) {
    		moves = new ArrayList<Move>();
    		int roleIndex = machine.rolesIndexMap[i];
    		int nextRoleIndex = machine.rolesIndexMap[i + 1];

    		for (int j = roleIndex; j < nextRoleIndex; ++j) {
    			if (currLegals.fastGet(j)) {
    				moves.add(machine.legalArray[j]);
    			}
    		}
    		randomJointMove.add(moves.get(rand.nextInt(moves.size())));
    	}

    	int start = machine.rolesIndexMap[size];
    	int end = machine.legalArray.length;
    	moves = new ArrayList<Move>();
    	for(int i = start; i < end; ++i) {
    		if (currLegals.fastGet(i)) {
    			moves.add(machine.legalArray[i]);
    		}
    	}
    	randomJointMove.add(moves.get(rand.nextInt(moves.size())));
        return randomJointMove;

    }

    public Move getRandomMove(OpenBitSet state, int rIndex) throws MoveDefinitionException
    {
        List<Move> legals = getLegalMoves(state, rIndex);
        return legals.get(rand.nextInt(legals.size()));
    }

    /**
     * Returns a random joint move from among all the possible joint moves in
     * the given state in which the given role makes the given move. Assumes move is a valid move
     */
    public List<Move> getRandomJointMove(OpenBitSet state, int rIndex, Move move) throws MoveDefinitionException
    {
    	List<List<Move>> randJointMove = new ArrayList<List<Move>>();
    	for (List<Move> jointMove : getLegalJointMoves(state)) {
    		if (jointMove.get(rIndex).equals(move)) randJointMove.add(jointMove);
    	}
    	return randJointMove.get(rand.nextInt(randJointMove.size()));//ASSUMES move is a valid move
    }

    @Override
	public OpenBitSet getRandomNextState(OpenBitSet state) throws MoveDefinitionException, TransitionDefinitionException
    {
        List<Move> random = getRandomJointMoveCurr(state);
        return getNextStateCurr(state, random);
    }

    /**
     * Computes all possible actions for role.
     */
    @Override
    public List<Move> findActions(Role role)
            throws MoveDefinitionException {
    	return machine.actions.get(role);
    }

    @Override
    public XPropNet getPropNet() {
    	return machine.propNet;
    }


	public List<Move> getLegalMoves(OpenBitSet state, int rIndex)//Change such that we don't have to keep updating legal moves
            throws MoveDefinitionException {
    	setState(state, null);

    	List<Move> moves = new ArrayList<Move>();
    	int roleIndex = machine.rolesIndexMap[rIndex];
    	int nextRoleIndex = (rIndex == (machine.roles.length - 1) ? machine.legalArray.length : machine.rolesIndexMap[rIndex + 1]);
    	for (int i = roleIndex; i < nextRoleIndex; ++i) {
			if (currLegals.fastGet(i)) {
				moves.add(machine.legalArray[i]);
			}
		}

    	return moves;
    }



	protected void setBases(OpenBitSet state) {
    	if (state == null) return;
    	int[] bases = machine.propNet.getBasePropositions();
    	int size = bases.length;

    	OpenBitSet temp = (OpenBitSet) state.clone();
    	state.xor(currentState);
    	currentState = temp;

    	for (int i = state.nextSetBit(0); i != -1; i = state.nextSetBit(i + 1)) {
    		boolean val = temp.fastGet(i);
    		if (val) components[i] += 1;
    		else components[i] -= 1;

    		long comp = compInfo[i];
    		int num_outputs = (int) ((comp & OUTPUTS_MASK) >> OUTPUT_SHIFT);
        	int outputsIndex = (int) (comp & OFFSET_MASK);

        	for (int j = 0; j < num_outputs; ++j) {
        		int outIndex = connecTable[outputsIndex + j];
        		int outValue = components[outIndex];
        		int lastPropagatedValue = (outValue & CURR_VAL_MASK);
        		if (val) components[outIndex] += 1;
        		else components[outIndex] -= 1;

        		int newVal = (components[outIndex] & CURR_VAL_MASK);
        		if (newVal != lastPropagatedValue) {
        			q.add(newVal | outIndex);
        		}
        	}
    	}

    }


	protected void setActions(OpenBitSet moves) {
    	if(moves == null) return;

    	int[] inputs = machine.propNet.getInputPropositions();
    	int size = inputs.length;

    	OpenBitSet tempInputs = (OpenBitSet) moves.clone();
    	moves.xor(currInputs);
    	currInputs = tempInputs;

    	for (int i = moves.nextSetBit(0); i != -1; i = moves.nextSetBit(i + 1)) {
    		boolean val = currInputs.fastGet(i);
    		int inputIndex = machine.inputOffset + i;
    		if (val) components[inputIndex] += 1;
    		else components[inputIndex] -= 1;

    		long comp = compInfo[inputIndex];
    		int num_outputs = (int) ((comp & OUTPUTS_MASK) >> OUTPUT_SHIFT);
        	int outputsIndex = (int) (comp & OFFSET_MASK);

        	for (int j = 0; j < num_outputs; ++j) {
        		int outIndex = connecTable[outputsIndex + j];

        		int lastPropagatedValue = (components[outIndex] & CURR_VAL_MASK);
        		if (val) components[outIndex] += 1;
        		else components[outIndex] -= 1;

        		int newVal = (components[outIndex] & CURR_VAL_MASK);
        		if (newVal != lastPropagatedValue) {
        			q.add(newVal | outIndex);
        		}
        	}
    	}

    }

	protected OpenBitSet movesToBit(List<Move> moves) {
		if (moves == null || moves.isEmpty()) return null;

		OpenBitSet movesSet = new OpenBitSet(machine.numInputs);
		for (int i = 0; i < machine.roles.length; ++i) {
			int index = machine.roleMoves.get(moves.get(i))[i];
			movesSet.fastSet(index);
		}

		return movesSet;
	}

	protected void setState(OpenBitSet state, List<Move> moves) {
    	setBases((OpenBitSet)state.clone());
    	setActions(movesToBit(moves));
    	propagate();
    }

	protected void setStateCurr(OpenBitSet state, List<Move> moves) {
    	//setBases((OpenBitSet)state.clone());
    	setActions(movesToBit(moves));
    	propagate();
    }


    @Override
	public OpenBitSet getNextState(OpenBitSet state, List<Move> moves) {

    	setState(state, moves);

    	return (OpenBitSet) nextState.clone();
    }

    public OpenBitSet getNextStateCurr(OpenBitSet state, List<Move> moves) {

    	setStateCurr(state, moves);

    	return (OpenBitSet) nextState.clone();
    }

	protected void resetPropNet() {
    	currInputs = new OpenBitSet(machine.numInputs);
    	currentState = new OpenBitSet(machine.numBases);
   		currLegals = new OpenBitSet(machine.numLegals);
   		nextState = new OpenBitSet(machine.numBases);
   		components = machine.propNet.getComponents();
    }

    @Override
    public boolean isTerminal(OpenBitSet state) {
    	setState(state, null);
    	return (components[machine.propNet.getTerminalProposition()] & CURR_VAL_MASK) != 0;
    }

    //goal Propositions will never be Trigger components, so we
    //can use its 2nd bit. Goal value is stored in bits 2-8, reading
    //from the left
    private static final long GOAL_MASK = 0x7F_0000_0000_000000L;
    private static final int GOAL_SHIFT = TYPE_SHIFT;

	protected int getGoalValue(long value) {//inline
    	return (int) ((value & GOAL_MASK) >> TYPE_SHIFT);
    }

    @Override
	public List<Integer> getGoals(OpenBitSet state) throws GoalDefinitionException {
        List<Integer> theGoals = new ArrayList<Integer>();
        for (int i = 0; i < machine.roles.length; ++i) {
            theGoals.add(getGoal(state, i));
        }
        return theGoals;
    }


	public int getGoal(OpenBitSet state, int rIndex)
            throws GoalDefinitionException {

    	setState(state, null);
        int[] rewards = machine.goalPropositions[rIndex];
        int size = rewards.length;

        for(int i = 0; i < size; ++i) {
        	int rewardIndex = rewards[i];
        	int value = components[rewardIndex];
        	if ((value & CURR_VAL_MASK) != 0) {
        		int goalVal = (int) ((compInfo[rewardIndex] & GOAL_MASK) >> TYPE_SHIFT);
        		return goalVal;
        	}

        }
        System.out.println("ERROR! Reward not defined in state " + state.toString());
        System.exit(0);
        return 0;
    }

    /* Already implemented for you */
    @Override
    public List<Role> getRoles() {
    	List<Role> rs = new ArrayList<Role>();
        for (int i = 0; i < machine.roles.length; ++i) rs.add(machine.roles[i]);
        return rs;
    }

    @Override
	public MachineState toGdl(OpenBitSet state) {
    	Set<GdlSentence> bases = new HashSet<GdlSentence>();
    	for (int i = state.nextSetBit(0); i != -1; i = state.nextSetBit(i + 1)) {
    		bases.add((GdlSentence) machine.gdlSentenceMap.get(i));
    	}
    	return new MachineState(bases);
    }

    @Override
	public OpenBitSet toBit(MachineState state) {
    	Set<GdlSentence> bases = state.getContents();
    	HashMap<GdlSentence, Integer> basesMap = machine.propNet.getBasesMap();
    	OpenBitSet bitSet = new OpenBitSet(machine.numBases);
    	for (GdlSentence base : bases) {
    		bitSet.fastSet(basesMap.get(base));
    	}
    	return bitSet;
    }

	@Override
	public MachineState getMachineStateFromBitSet(OpenBitSet contents) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<Move> getLegalMoves(OpenBitSet state, Role role)
			throws MoveDefinitionException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getGoal(OpenBitSet state, Role role)
			throws GoalDefinitionException {
		// TODO Auto-generated method stub
		System.out.println("Shouldn't call this method");
		System.exit(0);
		return 0;
	}

	public double getCurrGoal(OpenBitSet state, int rIndex)
            throws GoalDefinitionException {

    	//setState(state, null);
        int[] rewards = machine.goalPropositions[rIndex];
        int size = rewards.length;

        for(int i = 0; i < size; ++i) {
        	int rewardIndex = rewards[i];
        	int value = components[rewardIndex];
        	if ((value & CURR_VAL_MASK) != 0) {
        		int goalVal = (int) ((compInfo[rewardIndex] & GOAL_MASK) >> TYPE_SHIFT);
        		return (double)goalVal;
        	}

        }
        System.out.println("ERROR! Curr Reward not defined in state " + state.toString());
        System.exit(0);
        return 0;
    }

	public double[] getCurrGoal(OpenBitSet state)
            throws GoalDefinitionException {

		double[] rewards = new double[machine.roles.length];
		for (int i = 0; i < machine.roles.length; ++i) {
			rewards[i] = getCurrGoal(state, i);
		}
		return rewards;
	}

	public OpenBitSet getNextStateBit(OpenBitSet state, OpenBitSet moves) {

		setBases((OpenBitSet)state.clone());
    	setActions(moves);
    	propagate();

    	return (OpenBitSet) nextState.clone();
    }

	public OpenBitSet getRandomNextStateBit(OpenBitSet state) throws MoveDefinitionException, TransitionDefinitionException
    {
        OpenBitSet random = getRandomJointMoveBit(state);
        setBases((OpenBitSet)state.clone());
    	setActions(random);
    	propagate();

    	return (OpenBitSet) nextState.clone();
    }

	public OpenBitSet getRandomJointMoveBit(OpenBitSet state)
    {
    	setState(state, null);

    	int size = machine.roles.length;
    	OpenBitSet nextLegals = new OpenBitSet(machine.numLegals);
    	List<Integer> moves;

    	for (int i = 0; i < size; ++i) {
    		moves = new ArrayList<Integer>();
    		int roleIndex = machine.rolesIndexMap[i];
    		int nextRoleIndex = machine.rolesIndexMap[i + 1];

    		for (int j = roleIndex; j < nextRoleIndex; ++j) {
    			if (currLegals.fastGet(j)) {
    				moves.add(j);
    			}
    		}
    		nextLegals.fastSet(moves.get(rand.nextInt(moves.size())));
    	}

        return nextLegals;

    }

	public double[] Playout(XNodeAbstract n) throws MoveDefinitionException, TransitionDefinitionException, GoalDefinitionException {
		OpenBitSet state = n.state;
		while(!isTerminal(state)) {
			state = getRandomNextState(state);
		}
		return getCurrGoal(state);
	}


	@Override
	public void initialize(List<Gdl> description) {
		// TODO Auto-generated method stub
		return;
	}


}

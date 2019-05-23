package org.ggp.base.util.propnet.architecture;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Stack;

import org.ggp.base.util.Pair;
import org.ggp.base.util.gdl.grammar.GdlConstant;
import org.ggp.base.util.gdl.grammar.GdlRelation;
import org.ggp.base.util.gdl.grammar.GdlSentence;
import org.ggp.base.util.gdl.grammar.GdlTerm;
import org.ggp.base.util.logging.GamerLogger;
import org.ggp.base.util.propnet.architecture.components.And;
import org.ggp.base.util.propnet.architecture.components.Constant;
import org.ggp.base.util.propnet.architecture.components.Not;
import org.ggp.base.util.propnet.architecture.components.Proposition;
import org.ggp.base.util.propnet.architecture.components.Transition;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.Role;

import cern.colt.map.OpenIntObjectHashMap;



public final class XPropNet implements Serializable
{

	private static final long serialVersionUID = -7208637729754171622L;

	private static final int NUM_TYPE_BITS = 8;
	private static final int NUM_INPUT_BITS = 16;
	private static final int NUM_OUTPUT_BITS = 16;
	private static final int NUM_OFFSET_BITS = 24;

	private static final int OUTPUT_SHIFT = NUM_OFFSET_BITS;
	private static final int INPUT_SHIFT = OUTPUT_SHIFT + NUM_OUTPUT_BITS;
	private static final int TYPE_SHIFT = INPUT_SHIFT + NUM_INPUT_BITS;


	private static final long NOT_TRIGGER = 0;
	private static final long TRIGGER_TRANSITION = 0xC0_0000_0000_000000L;
	private static final long TRIGGER_LEGAL = 0x80_0000_0000_000000L;
	private static final int INIT_TRUE = 0x8000_0000;
	private static final int INIT_FALSE = 0;
	private static final int INIT_NOT = 0xFFFF_FFFF;
	private static final int INIT_DEFAULT = 0x7FFF_FFFF;

	private int numBases, baseOffset, numLegals, numInputs, legalOffset, inputOffset;
	private final Role[] roles;
	private HashMap<Role, Integer> roleMap;
	//private TCustomHashMap<Role,int> roleMap;
	private int initProposition, terminalProposition;
    private int[] basePropositions;
    private int[] inputPropositions;
    private int[] constants;
    private int[] components;
    private int[] initBases;
    private long[] compInfo;
    private int[] connecTable;
    private int[][] goalPropositions;
    private HashMap<Role, List<Move>> actionsMap;
    private OpenIntObjectHashMap gdlSentenceMap;
    private HashMap<GdlSentence, Integer> basesMap;
    private int[] rolesIndexMap;
    private OpenIntObjectHashMap indexCompMap;
    private HashMap<Pair<Role, Move>, Integer> legalMoveMap;
    private Move[] legalArray;
    private HashMap<Move, int[]> roleMoves;
    private int[] ordering;

    public HashMap<Component, List<Component>> outputMap;
    PropNet oldProp;
    HashMap<Component, Integer> compIndexMap;

    private static final long OFFSET_MASK = 0x00_0000_0000_FFFFFFL;

    protected int outputsOffset(long comp) {
    	return (int) (comp & OFFSET_MASK);
    }

	@SuppressWarnings("unused")
	public XPropNet(PropNet prop)
	{
		System.out.println("XPropNet initializing...");

		//test factoring
		//prop = PropNet.factor_propnet(prop, prop.getRoles().get(0)).get(0);


		//prop.renderToFile("preOpt.dot");
		optimizeProp(prop);
		//prop.renderToFile("postOpt.dot");

		int nLegals = 0;
		int nInputs = prop.getInputPropositions().keySet().size();
		for (Entry<Role, Set<Proposition>> e : prop.getLegalPropositions().entrySet()) {
			nLegals += e.getValue().size();
		}
		System.out.println("NumLegals: " + nLegals + " NumInputs: " + nInputs);

		/*for (Proposition i : prop.getInputPropositions().values()) {
			if (prop.getLegalInputMap().get(i) == null) {
				System.out.println(i.toString());
				System.out.println("   Outputs:");
				for (Component out : i.getOutputs()) System.out.println("    " + out.toString());
			}
		}*/
		//if (nLegals != nInputs) System.exit(0);

		oldProp = prop;
		Set<Component> pComponents = prop.getComponents();
	    roles = prop.getRoles().toArray(new Role[prop.getRoles().size()]);
	    int size = roles.length;
	    roleMap = new HashMap<Role, Integer>();
	    for (int i = 0; i < size; ++i) {
	    	roleMap.put(roles[i], i);
	    }

		Map<Role, Set<Proposition>> moveMap = prop.getLegalPropositions();
		//Mapping from component to component ID
		HashMap<Component, Integer> compIndices = new HashMap<Component, Integer>();
		int compId = 0, total_outputs = 0;

		//Components that we have already processed
		HashSet<Component> props = new HashSet<Component>(prop.getBasePropositions().values());
		props.addAll(prop.getInputPropositions().values());

/*
 * Define Proposition ordering for Bases. Populate gdlSentenceMap (mapping from GdlSentence to component ID) and
 * basesMap (mapping from component ID to GdlSentence for bases). Set numBases and baseOffset into components
 * and compInfo array. Define component IDs for bases and increment total outputs.
 */
		List<Proposition> bases = new ArrayList<Proposition>();
		numBases = 0; baseOffset = 0;
		gdlSentenceMap = new OpenIntObjectHashMap();//Mapping from compId to GdlSentence for bases
		basesMap = new HashMap<GdlSentence, Integer>();//Mapping from GdlSentence to compId
		for (Entry<GdlSentence, Proposition> e : prop.getBasePropositions().entrySet()) {
			GdlSentence s = e.getKey();
			Proposition b = e.getValue();

			gdlSentenceMap.put(compId, s);
			basesMap.put(s, compId - baseOffset);
			compIndices.put(b, compId);
			++compId;
			bases.add(b);
			++numBases;
			total_outputs += b.getOutputs().size();
		}


/*
 * Define Proposition ordering for Legals. Set numLegals and legalOffset.
 */


		List<List<Proposition>> legals  = new ArrayList<List<Proposition>>();
		List<Proposition> inputs = new ArrayList<Proposition>();
		Map<Proposition, Proposition> legalInputMap = prop.getLegalInputMap();
		List<Move> legalArr = new ArrayList<Move>();//List of all moves in the game, in order of role
		actionsMap = new HashMap<Role, List<Move>>();
		rolesIndexMap = new int[roles.length + 1];
		numLegals = 0; legalOffset = compId;
		for (int i = 0; i < roles.length; ++i) {
			List<Proposition> rLegals = new ArrayList<Proposition>(moveMap.get(roles[i]));
			rolesIndexMap[i] = compId - legalOffset;
			List<Move> rMoves = new ArrayList<Move>();

			for (Proposition l : rLegals) {
				Move m = new Move(l.getName().getBody().get(1));
				legalArr.add(m);
				rMoves.add(m);
				compIndices.put(l, compId);
				++compId;
				total_outputs += l.getOutputs().size();
				if (legalInputMap.get(l) != null) {
					inputs.add(legalInputMap.get(l));
				}

			}

			actionsMap.put(roles[i], rMoves);
			numLegals += rLegals.size();
			props.addAll(rLegals);
			legals.add(rLegals);
		}

		for (Proposition input : prop.getInputPropositions().values()) {
			if (inputs.contains(input)) {
				continue;
			}
			inputs.add(input);
		}

		legalArray = legalArr.toArray(new Move[legalArr.size()]);
		rolesIndexMap[roles.length] = legalArray.length;


/*
 * Define Proposition ordering from Inputs. Set numInputs and inputOffset
 */

		numInputs = 0; inputOffset = compId;
		for (Proposition i : inputs) {
			if (i != null) {
				compIndices.put(i, compId);
				++compId;
				++numInputs;
				total_outputs += i.getOutputs().size();
			}
		}
		props.addAll(inputs);



/*
 * Everything Else
 */
///////////////////////////

		Proposition init = prop.getInitProposition();
		if (init == null) initProposition = -1;
		Proposition term = prop.getTerminalProposition();
		outputMap = new HashMap<Component, List<Component>>();
		for (Component c : pComponents) {
			outputMap.put(c, new ArrayList<Component>(c.getOutputs()));
			if (c.equals(init)) initProposition = compId;
			if (c.equals(term)) terminalProposition = compId;

			if (!props.contains(c)) {
				props.add(c);
				compIndices.put(c, compId);
				++compId;
				total_outputs += c.getOutputs().size();

			}
		}

		compIndexMap = compIndices;
		indexCompMap = new OpenIntObjectHashMap();
		for (Component c : compIndexMap.keySet()) indexCompMap.put(compIndexMap.get(c).intValue(), c);


//Add bases, inputs, legals to props
		props = new HashSet<Component>(prop.getBasePropositions().values());
		props.addAll(prop.getInputPropositions().values());
		for (int i = 0; i < roles.length; ++i) {
			props.addAll(legals.get(i));
		}

		connecTable = new int[total_outputs];
		System.out.println("connecTableLength: " + connecTable.length);
		components = new int[pComponents.size()];
		compInfo = new long[pComponents.size()];
		int outputIndex = 0;


/*
 * BASES
 */
		basePropositions = new int[numBases];
		for (int i = 0; i < numBases; ++i) {
			basePropositions[i] = baseOffset + i;
			Proposition b = bases.get(i);
			if (compIndices.get(b) != (baseOffset + i)) {
				System.out.println("compIndices.get(b) != (baseOffset + i)");
				System.exit(0);
			}
		}

		for (Proposition b : bases) {
			long type = NOT_TRIGGER;
			long num_inputs = ((long)b.getInputs().size()) << INPUT_SHIFT;
			List<Component> outputs = outputMap.get(b);
			long num_outputs = ((long)outputs.size()) << OUTPUT_SHIFT;
			long outIndex = ((long)outputIndex);
			long info = type | num_inputs | num_outputs | outIndex;
			compInfo[compIndices.get(b)] = info;
			components[compIndices.get(b)] = INIT_DEFAULT;
			for (Component out : outputs) {
				connecTable[outputIndex] = compIndices.get(out);
				++outputIndex;
			}
		}


		legalMoveMap = new HashMap<Pair<Role, Move>, Integer>();

/*
 * LEGALS
 */

		for (int i = 0; i < roles.length; ++i) {
			List<Proposition> ls = legals.get(i);
			props.addAll(ls);

			for (Proposition l : ls) {

				Pair<Role, Move> p = Pair.of(roles[i], new Move(l.getName().getBody().get(1)));
				legalMoveMap.put(p, compIndices.get(l));

				long type = TRIGGER_LEGAL;
				long num_inputs = ((long)l.getInputs().size()) << INPUT_SHIFT;
				List<Component> outputs = outputMap.get(l);
				long num_outputs = ((long)outputs.size()) << OUTPUT_SHIFT;
				long outIndex = ((long)outputIndex);
				long info = type | num_inputs | num_outputs | outIndex;
				compInfo[compIndices.get(l)] = info;
				components[compIndices.get(l)] = INIT_DEFAULT;
				for (Component out : outputs) {
					connecTable[outputIndex] = compIndices.get(out);
					++outputIndex;
				}
			}
		}


/*
 * INPUTS
 */
		inputPropositions = new int[numInputs];
		for (int i = 0; i < numInputs; ++i) {
			inputPropositions[i] = inputOffset + i;
			Proposition in = inputs.get(i);
			if (in != null) {
				if (compIndices.get(in) != (inputOffset + i)) {
					System.out.println("compIndices.get(in) != (inputOffset + i)");
					//System.exit(0);
				}
			}
		}

		roleMoves = new HashMap<Move, int[]>();
		int[] pseudoNoops = new int[roles.length];
		for (Proposition i : inputs) {

			List<GdlTerm> iGdl = i.getName().getBody();
			Move m = new Move(iGdl.get(1));
			Role r = new Role((GdlConstant) iGdl.get(0));
			if(!roleMoves.containsKey(m)) {
				roleMoves.put(m, new int[roles.length]);
				Arrays.fill(roleMoves.get(m), -1);
			}

			int inIndex = compIndices.get(i) - inputOffset;
			roleMoves.get(m)[roleMap.get(r)] = inIndex;

			if (iGdl.get(1).toString().equals("noope")) {
				pseudoNoops[roleMap.get(r)] = inIndex;
			}

			long type = NOT_TRIGGER;
			long num_inputs = ((long)i.getInputs().size()) << INPUT_SHIFT;
			List<Component> outputs = outputMap.get(i);
			long num_outputs = ((long)outputs.size()) << OUTPUT_SHIFT;
			long outIndex = ((long)outputIndex);
			long info = type | num_inputs | num_outputs | outIndex;
			compInfo[compIndices.get(i)] = info;
			components[compIndices.get(i)] = INIT_DEFAULT;
			for (Component out : outputs) {
				connecTable[outputIndex] = compIndices.get(out);
				++outputIndex;
			}
		}

		for (int[] inIndexes : roleMoves.values()) {
			for (int i=0;i<inIndexes.length;++i) {
				if (inIndexes[i] == -1) {
					inIndexes[i] = pseudoNoops[i];
				}
			}
		}


/*
 * TRANSITIONS
 */
		for (Component c : pComponents) {
			if (c instanceof Transition) {
				props.add(c);
				long type = TRIGGER_TRANSITION;
				long num_inputs = ((long)c.getInputs().size()) << INPUT_SHIFT;
				List<Component> outputs = outputMap.get(c);
				long num_outputs = ((long)outputs.size()) << OUTPUT_SHIFT;
				long outIndex = ((long)outputIndex);
				long info = type | num_inputs | num_outputs | outIndex;
				compInfo[compIndices.get(c)] = info;
				components[compIndices.get(c)] = INIT_DEFAULT;
				for (Component out : outputs) {
					connecTable[outputIndex] = compIndices.get(out);
					++outputIndex;
				}

			}
		}

/*
 * GOAL PROPOSITIONS
 */
		Map<Role, Set<Proposition>> goalProps = prop.getGoalPropositions();
		goalPropositions = new int[roles.length][];
		for (int j = 0; j < roles.length; ++j) {
			Role r = roles[j];
			int[] rewards = new int[goalProps.get(r).size()];
			int i = 0;
			for (Proposition g : goalProps.get(r)) {
				props.add(g);
				GdlRelation relation = (GdlRelation) g.getName();
		        GdlConstant constant = (GdlConstant) relation.get(1);
		        int goalVal = Integer.parseInt(constant.toString());
				long type = ((long)goalVal) << TYPE_SHIFT;
				long num_inputs = ((long)g.getInputs().size()) << INPUT_SHIFT;
				List<Component> outputs = outputMap.get(g);
				long num_outputs = ((long)outputs.size()) << OUTPUT_SHIFT;
				long outIndex = ((long)outputIndex);
				long info = type | num_inputs | num_outputs | outIndex;
				compInfo[compIndices.get(g)] = info;
				components[compIndices.get(g)] = INIT_DEFAULT;
				rewards[i] = compIndices.get(g);
				++i;
				for (Component out : outputs) {
					connecTable[outputIndex] = compIndices.get(out);
					++outputIndex;
				}
			}
			goalPropositions[j] = rewards;
		}

/*
 * CONSTANTS & Everything Else
 */
		List<Integer> consts = new ArrayList<Integer>();
		for (Component c : pComponents) {
			if (c instanceof Constant) consts.add(compIndices.get(c));

			if (!props.contains(c)) {

				if (c instanceof And) {
					components[compIndices.get(c)] = INIT_TRUE - c.getInputs().size();
				} else if (c instanceof Not) {
					components[compIndices.get(c)] = INIT_NOT;
				} else if (c instanceof Constant) {
					components[compIndices.get(c)] = c.getValue() ? INIT_TRUE : INIT_FALSE;
				} else if (c.equals(init)) {
					if (compIndices.get(init) != initProposition) {
						System.out.println("init compId incorrect");
						System.exit(0);
					}
					components[compIndices.get(c)] = INIT_TRUE;
				} else {
					components[compIndices.get(c)] = INIT_DEFAULT;
				}

				long type = NOT_TRIGGER;
				long num_inputs = ((long)c.getInputs().size()) << INPUT_SHIFT;
				List<Component> outputs = outputMap.get(c);
				long num_outputs = ((long)outputs.size()) << OUTPUT_SHIFT;
				long outIndex = ((long)outputIndex);
				long info = type | num_inputs | num_outputs | outIndex;
				compInfo[compIndices.get(c)] = info;
				for (Component out : outputs) {
					connecTable[outputIndex] = compIndices.get(out);
					++outputIndex;
				}

			}

		}
		size = consts.size();
		constants = (consts.isEmpty() ? null : new int[size]);
		for (int i = 0; i < size; ++i) constants[i] = consts.get(i);

		checkPropNet();

		HashSet<Component> initB = new HashSet<Component>();
		Proposition initProp = prop.getInitProposition();
		if (initProp != null) {
			Stack<Component> s = new Stack<Component>();
			s.push(initProp);
			while (!s.isEmpty()) {
				Component c = s.pop();
				if (c instanceof Transition) {
					initB.add(c.getSingleOutput());
				} else {
					for (Component out : c.getOutputs()) {
						s.push(out);
					}
				}
			}

			initBases = new int[initB.size()];
			int index = 0;
			for (Component b : initB) {
				initBases[index++] = compIndexMap.get(b);
			}
		} else {
			initBases = null;
		}




/*
 * Compute topological ordering
 */
		Stack<Integer> ord = new Stack<Integer>();
    	HashSet<Component> visited = new HashSet<Component>();
    	Component initP = (Component) indexCompMap.get(initProposition);
    	initP = null;
    	if (initP != null) {
    		for (Component out : initP.getOutputs()) {
    			if (!visited.contains(out)) {
    				visited.add(out);
    				topologicalSort(out, ord, visited);
    			}
    		}
    	}

    	for (Component b : bases) {
    		for (Component out : b.getOutputs()) {
    			if (!visited.contains(out)) {
    				visited.add(out);
    				topologicalSort(out, ord, visited);
    			}
    		}
    	}

    	for (Component i : inputs) {
    		for (Component out : i.getOutputs()) {
    			if (!visited.contains(out)) {
    				visited.add(out);
    				topologicalSort(out, ord, visited);
    			}
    		}
    	}

    	for (int i = 0; i < size; ++i) {
    		Component c = (Component) indexCompMap.get(constants[i]);
    		for (Component out : c.getOutputs()) {
    			if (!visited.contains(out)) {
    				visited.add(out);
    				topologicalSort(out, ord, visited);
    			}
    		}
    	}

    	size = ord.size();
    	ordering = new int[size];
        for (int i = 0; i < size; ++i) {
        	int val = ord.pop().intValue();
            ordering[i] = val;
        }

	}


	public int[] initBases() {
		return initBases;
	}

	public int[] getComponents() {
		return components.clone();
	}

	public int[] getBasePropositions() {
		return basePropositions;
	}

	public int[] getInputPropositions() {
		return inputPropositions;
	}

	public int getInitProposition() {
		return initProposition;
	}

	public int getTerminalProposition() {
		return terminalProposition;
	}


	public int[][] getGoalPropositions() {
		return goalPropositions;
	}

	public HashMap<Pair<Role, Move>, Integer> getLegalMoveMap() {
		return legalMoveMap;
	}

	public long[] getCompInfo() {
		return compInfo;
	}

	public int[] getConnecTable() {
		return connecTable;
	}

	public Role[] getRoles() {
		return roles;
	}

	public int numBases() {
		return numBases;
	}

	public int numInputs() {
		return numInputs;
	}

	public int numLegals() {
		return numLegals;
	}

	public int getBaseOffset() {
		return baseOffset;
	}

	public int getInputOffset() {
		return inputOffset;
	}

	public int getLegalOffset() {
		return legalOffset;
	}

	public HashMap<Role, List<Move>> getActionsMap() {
		return actionsMap;
	}

	public Move[] getLegalArray() {
		return legalArray;
	}

	public int[] getConstants() {
		return constants;
	}

	public OpenIntObjectHashMap getGdlSentenceMap() {
		return gdlSentenceMap;
	}

	public HashMap<GdlSentence, Integer> getBasesMap() {
		return basesMap;
	}

	public int[] getRolesIndexMap() {
		return rolesIndexMap;
	}

	public HashMap<Move, int[]> getRoleMoves() {
		return roleMoves;
	}

	public OpenIntObjectHashMap indexCompMap() {
		return indexCompMap;
	}

	public HashMap<Component, Integer> compIndexMap() {
		return compIndexMap;
	}

	public int numOutputs(long comp) {//inline these functions
    	return (int) ((comp & 0x00_0000_FFFF_000000L) >> 24);
    }

    public int numInputs(long comp) {
    	return (int) ((comp & 0x00_FFFF_0000_000000L) >> 40);
    }

    public void checkPropNet() {
    	int numOutputsP = 0;
    	for ( Component component : compIndexMap.keySet())
		{
    		Set<Component> inputs = component.getInputs();
    		for (Component in : inputs) {
    			if (!(in.getOutputs().contains(component))) {
    				System.out.println("!(in.getOutputs_set().contains(component))");
    				System.out.println(component + " Inputs: " + inputs.toString());
    				System.out.println(in + "Outputs: " + in.getOutputs().toString());
    				System.exit(0);
    			}
    		}
    		numOutputsP += component.getOutputs().size();
			int index = compIndexMap.get(component);
			if (numInputs(compInfo[index]) != component.getInputs().size()) {
				System.out.println(component);
				System.out.println("NumInputs incorrect: " + "Correct: " + component.getInputs().size() + " Incorrect: " + numInputs(compInfo[index]));
				String hex = Long.toHexString(compInfo[index]);
				String pad = "";
				for (int i = 0; i < 16 - hex.length(); ++i) pad += "0";
				hex = "0x" + pad + hex;
				System.out.println("compInfo: " + hex);
				System.exit(0);
			}
			if (numOutputs(compInfo[index]) != component.getOutputs().size()) {
				System.out.println(component);
				System.out.println("NumOutputs incorrect: " + "Correct: " + component.getOutputs().size() + " Incorrect: " + component.getOutputs().size());
				String hex = Long.toHexString(compInfo[index]);
				String pad = "";
				for (int i = 0; i < 16 - hex.length(); ++i) pad += "0";
				hex = "0x" + pad + hex;
				System.out.println("compInfo: " + hex);
				System.exit(0);
			}
		}
    	if (connecTable.length != numOutputsP) {
    		System.out.println("connecTable.length != numOutputsP");
    		System.out.println(connecTable.length + ", " + numOutputsP);
    		System.exit(0);
    	}
		System.out.println("CORRECT!");

    }


    protected void topologicalSort(Component c, Stack<Integer> s, HashSet<Component> visited) {
    	if (!(c instanceof Transition)) {
    		for (Component out : c.getOutputs()) {
        		if (!visited.contains(out)) {
        			visited.add(out);
        			topologicalSort(out, s, visited);
        		}
        	}
    	}

    	s.push(compIndexMap.get(c));
    }

    @SuppressWarnings("unchecked")
	public int[] getOrdering() {
    	return ordering;
    }


	public String bitString(int[] comps)
	{
		StringBuilder sb = new StringBuilder();

		sb.append("digraph propNet\n{\n");
		for ( Component component : compIndexMap.keySet())
		{
			int index = compIndexMap.get(component);
			//sb.append("\t" + component.bitString(comps[index]) + "\n");
			sb.append("\t" + component.bitString(comps[index]) + "\n");
		}
		sb.append("}");

		return sb.toString();
	}

	/**
     * Outputs the propnet in .dot format to a particular file.
     * This can be viewed with tools like Graphviz and ZGRViewer.
     *
     * @param filename the name of the file to output to
     */
    public void renderToFile(String filename, int[] comps) {
        try {
            File f = new File(filename);
            FileOutputStream fos = new FileOutputStream(f);
            OutputStreamWriter fout = new OutputStreamWriter(fos, "UTF-8");
            fout.write(bitString(comps));
            fout.close();
            fos.close();
        } catch(Exception e) {
            GamerLogger.logStackTrace("StateMachine", e);
        }
        //oldProp.renderToFile("old" + filename);
    }

    protected void clearQ(ArrayDeque<Component> toRemove, PropNet prop) {
    	while (!toRemove.isEmpty()) {
    		Component c = toRemove.pop();
    		prop.removeComponent(c);
    	}
    }

    protected void optimizeProp(PropNet prop) {

    	int initSize = prop.getComponents().size();

    	/*
    	Map<Proposition, Proposition> legalInputMap = prop.getLegalInputMap();
    	HashSet<Proposition> is = new HashSet<Proposition>(prop.getInputPropositions().values());
    	for (Proposition i : is) {
    		if (legalInputMap.get(i) == null) {
    			prop.removeComponent(i);
    			for (Component out : i.getOutputs()) {
    				out.getInputs().remove(i);
    			}
    		}
    	}
    	*/

    	HashSet<Component> useful = new HashSet<Component>();
    	useful.addAll(prop.getLegalInputMap().keySet());
    	for (Entry<Role, Set<Proposition>> e : prop.getGoalPropositions().entrySet()) {
    		useful.addAll(e.getValue());
    	}
    	useful.add(prop.getTerminalProposition());
    	useful.addAll(prop.getBasePropositions().values());
    	useful.add(prop.getInitProposition());



    	for (Iterator<Proposition> it = prop.getPropositions().iterator(); it.hasNext();)  {
    		Proposition p = it.next();
    		if (useful.contains(p)) continue;

    		Set<Component> inputs = p.getInputs();
    		Set<Component> outputs = p.getOutputs();

    		if (inputs.size() == 1 && !outputs.isEmpty()) {
    			Component in = inputs.iterator().next();
    			if (in instanceof Transition) continue;

    			int presize = prop.getComponents().size();
    			it.remove();
    			prop.removeComponent(p);
    			for (Component out : outputs) {
    				out.addInput(in);
    				in.addOutput(out);
    			}
    			int postsize = prop.getComponents().size();
    			if (presize == postsize) {
    				System.out.println("presize == postsize");
    				System.exit(0);
    			}
    		}
    	}



    	int postSize = prop.getComponents().size();
    	System.out.println("NumComponentsInitial: " + initSize + " NumComponentFinal: " + postSize);
    	System.out.println("Savings: " + (100 * (initSize - postSize) / ((double)initSize)) + "%");

    }

}


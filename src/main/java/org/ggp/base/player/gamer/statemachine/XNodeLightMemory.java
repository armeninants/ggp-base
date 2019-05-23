package org.ggp.base.player.gamer.statemachine;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.lucene.util.OpenBitSet;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.XStateMachine;
import org.ggp.base.util.statemachine.exceptions.MoveDefinitionException;

public class XNodeLightMemory extends XNodeAbstract implements Serializable {

	private static final long serialVersionUID = -8233477291312873815L;
	public XNodeLightMemory(OpenBitSet state, double visits, double updates, double[] utility) {
		this.state = state;
		this.children = new ConcurrentHashMap<List<Move>, XNodeLightMemory>();
		this.legalJointMoves = new ConcurrentHashMap<Move, List<List<Move>>>();

		this.utility = utility;
		this.visits = visits;
		this.updates = updates;
		//this.sum_x = 0;
		//this.sum_x2 = 0;
		//this.n = 0;
		//this.C_CONST = 60;
		//this.expanded = false;
	}

	public XNodeLightMemory(OpenBitSet state, int numRoles) {
		this(state, 0., 0., new double[numRoles]);
	}
	public volatile double[] utility;
	public volatile double visits;
	public volatile double updates;
	//public volatile boolean expanded;
	//public volatile double sum_x;
	//public volatile double sum_x2;
	//public volatile int n;
	//public volatile double C_CONST;
	//dont serialize these
	//private transient volatile AtomicBoolean started = new AtomicBoolean(false);
	private transient volatile Map<List<Move>, XNodeLightMemory> children;
	private transient volatile Map<Move, List<List<Move>>> legalJointMoves;
	private transient volatile Move[] legalMoves;

	/*public AtomicBoolean isStarted(XStateMachine machine, int index) throws MoveDefinitionException {
		if (started == null) {
			reinitializeTransient(machine, index);
		}
		return started;
	}*/

	public Map<List<Move>, XNodeLightMemory> getChildren(XStateMachine machine, int index) throws MoveDefinitionException {
		if (children == null) {
			reinitializeTransient(machine, index);
		}
		return children;
	}

	public Map<Move, List<List<Move>>> getLegalJointMoves(XStateMachine machine, int index) throws MoveDefinitionException {
		if (legalJointMoves == null) {
			reinitializeTransient(machine, index);
		}
		return legalJointMoves;
	}

	public Move[] getLegalMoves(XStateMachine machine, int index) throws MoveDefinitionException {
		if (legalMoves == null) {
			reinitializeTransient(machine, index);
		}
		return legalMoves;
	}

	public void setLegalMoves(Move[] moves) {
		legalMoves = moves;
	}


	private void reinitializeTransient(XStateMachine machine, int index) throws MoveDefinitionException {

		//reinitialize transient elements
		List<Move> legalMovesList = machine.getLegalMoves(state, index);
		legalMoves = legalMovesList.toArray(new Move[legalMovesList.size()]);

		legalJointMoves = new HashMap<Move, List<List<Move>>>();
		for (Move move : legalMoves) {
			List<List<Move>> legalJointMovesList = machine.getLegalJointMoves(state, index, move);
			legalJointMoves.put(move, legalJointMovesList);
		}
		children = new HashMap<List<Move>, XNodeLightMemory>();
	}

}
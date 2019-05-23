package org.ggp.base.player.gamer.statemachine;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.lucene.util.OpenBitSet;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.exceptions.MoveDefinitionException;

public class XNodeLight extends XNodeAbstract {

	private static final long serialVersionUID = -8233477291312873815L;
	public XNodeLight(OpenBitSet state, double visits, double updates, double[] utility) {
		this.state = state;
		this.children = new ConcurrentHashMap<List<Move>, XNodeLight>();
		this.legalJointMoves = new ConcurrentHashMap<Move, List<List<Move>>>();

		this.utility = utility;
		this.visits = visits;
		this.updates = updates;
	}

	public XNodeLight(OpenBitSet state, int numRoles) {
		this(state, 0., 0., new double[numRoles]);
	}
	public volatile double[] utility;
	public volatile double visits;
	public volatile double updates;
	private volatile Map<List<Move>, XNodeLight> children;
	private volatile Map<Move, List<List<Move>>> legalJointMoves;
	private volatile Move[] legalMoves;


	public Map<List<Move>, XNodeLight> getChildren(int index) throws MoveDefinitionException {
		return children;
	}

	public Map<Move, List<List<Move>>> getLegalJointMoves(int index) throws MoveDefinitionException {
		return legalJointMoves;
	}

	public Move[] getLegalMoves(int index) throws MoveDefinitionException {
		return legalMoves;
	}

	public void setLegalMoves(Move[] moves) {
		legalMoves = moves;
	}

}
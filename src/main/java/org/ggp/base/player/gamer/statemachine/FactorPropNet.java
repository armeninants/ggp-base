package org.ggp.base.player.gamer.statemachine;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.ggp.base.util.Pair;
import org.ggp.base.util.gdl.grammar.GdlConstant;
import org.ggp.base.util.gdl.grammar.GdlPool;
import org.ggp.base.util.gdl.grammar.GdlRelation;
import org.ggp.base.util.gdl.grammar.GdlTerm;
import org.ggp.base.util.propnet.architecture.Component;
import org.ggp.base.util.propnet.architecture.PropNet;
import org.ggp.base.util.propnet.architecture.components.Constant;
import org.ggp.base.util.propnet.architecture.components.Or;
import org.ggp.base.util.propnet.architecture.components.Proposition;
import org.ggp.base.util.propnet.architecture.components.Transition;
import org.ggp.base.util.statemachine.Role;

import com.google.common.collect.ImmutableList;

public class FactorPropNet {

	PropNet prop;
	Role r;

	public FactorPropNet(PropNet prop, Role r) {
		this.prop = prop;
		this.r = r;
	}

	public List<PropNet> factor() {
		System.out.println("Initializing Factor Sequence..");
		//prop.renderToFile("unfactoredprop.dot");

		int PROP_SIZE_LIMIT = 10000;
		if (prop.getComponents().size() > PROP_SIZE_LIMIT) {
			List<PropNet> factoredProps = new ArrayList<>();
			PropNet prop2 = new PropNet(prop.getRoles(), prop.getComponents());
			factoredProps.add(prop2);
			System.out.println("Propnet too large to factor, gtfo. " + prop.getComponents().size());
			return factoredProps;
		}

		int cId = 0;
		HashMap<Component, Integer> cMap = new HashMap<Component, Integer>();
		HashMap<Integer, Component> iMap = new HashMap<Integer, Component>();
		Set<Component> constans = new HashSet<Component>();
		for (Component c : prop.getComponents()) {
			iMap.put(cId, c);
			cMap.put(c, cId++);
			if (c instanceof Constant) constans.add(c);
		}




		//label direct connections of components
		int[][] dependencyMatrix = initDependencyMatrix(prop.getComponents().size());
		for (Component c : prop.getComponents()) {
			int cIndex = cMap.get(c);
			Set<Component> outputs = c.getOutputs();
			for (Component out : outputs) {
				int outIndex = cMap.get(out);
				if (cIndex < outIndex) {
					dependencyMatrix[cIndex][outIndex] = 1;
				} else {
					dependencyMatrix[outIndex][cIndex] = 1;
				}
			}
		}

		//find the disjoint sets of connected components
		List<Set<Component>> disjointSets = new LinkedList<Set<Component>>();
		for(Component c : prop.getComponents()) {
			Set<Component> singleSet = new HashSet<Component>();
			singleSet.add(c);
			disjointSets.add(singleSet);
		}
		for(int i=0;i<dependencyMatrix.length;++i) {
			Set<Component> iSet = findSet(disjointSets, iMap.get(i));
			for(int j=i+1;j<dependencyMatrix[0].length;++j) {
				if (dependencyMatrix[i][j] == 1) {
	    			Set<Component> jSet = findSet(disjointSets, iMap.get(j));
	    			if (iSet.equals(jSet)) {
	    				continue;
	    			}
	    			iSet.addAll(jSet);
	    			disjointSets.remove(jSet);
				}
			}
		}
		System.out.println("Number of disjoint sets: " + disjointSets.size());
		/*for (Set<Component> s : disjointSets) {
			System.out.println(s.size());
		}*/

		//gather the noops
		Set<Component> noops = new HashSet<Component>();
		Iterator<Set<Component>> it = disjointSets.iterator();
		while (it.hasNext()) {
			Set<Component> disjointSet = it.next();
			if (disjointSet.size() == 1) { //houston, we have a noop?
				for (Component c : disjointSet) {
					//noops.add(prop.getLegalInputMap().get(c));
					noops.add(c);
					List<GdlTerm> gdlTerms = prop.getLegalInputMap().get(c).getName().getBody();
					for (GdlTerm term : gdlTerms) {
						System.out.println(term.toString());
						GdlTerm t = GdlPool.getConstant(term.toString());
						System.out.println(t.toString());
					}
					System.out.println("Found noop: " + prop.getLegalInputMap().get(c));
				}
				it.remove();
			}
		}


		//create a map for the best goal containing a connected input with its graph distance
		Set<Proposition> goals = prop.getGoalPropositions().get(r);
		Pair<Proposition, Integer> bestGoal = Pair.of(null, -1);
		for (Proposition g : goals) {
			GdlRelation relation = (GdlRelation) g.getName();
	        GdlConstant constant = (GdlConstant) relation.get(1);
	        int goalVal = Integer.parseInt(constant.toString());
			if (goalVal > bestGoal.right) {
				bestGoal = Pair.of(g, goalVal);
			}
		}
		System.out.println("Goal: " + bestGoal.left.getName());
		Map<Component, Integer> goalDependency = new HashMap<>();
		Set<Component> seen = new HashSet<>();
		inputDFS(bestGoal.left, 0, goalDependency, seen, prop, bestGoal.left.toString());
		//do the same for terminal state, bc it seems like a good idea
		inputDFS(prop.getTerminalProposition(), 0, goalDependency, seen, prop, prop.getTerminalProposition().toString());
		for (Component noop : noops) {
			goalDependency.put(noop, -1);
		}


		//remove sets which have no goal value > 0
		it = disjointSets.iterator();
		while (it.hasNext()) {
			Set<Component> disjointSet = it.next();
			boolean containsGoal = false;
			for (Proposition g : goals) {
				GdlRelation relation = (GdlRelation) g.getName();
		        GdlConstant constant = (GdlConstant) relation.get(1);
		        int goalVal = Integer.parseInt(constant.toString());
				if (goalVal == 0) {
					continue;
				}
				if (disjointSet.contains(g)) {
					containsGoal = true;
					break;
				}
			}

			if (!containsGoal) {
				it.remove();
			}
		}
		System.out.println("Number of disjoint sets with goals: " + disjointSets.size());
		System.out.println("disjoint size: " + disjointSets.get(0).size() + " total size: " + prop.getComponents().size());



		//get input props from the disjoint sets
		List<Proposition> inputProps = new ArrayList<Proposition>(prop.getInputPropositions().values());
		List<Set<Component>> disjointInputs = new ArrayList<Set<Component>>();
		for (Set<Component> disjointSet : disjointSets) {
			Set<Component> disjointInput = new HashSet<Component>();
			for (Proposition p : inputProps) {
				if (disjointSet.contains(p)) {
					disjointInput.add(p);
				}
			}
			disjointInput.addAll(noops);
			disjointInputs.add(disjointInput);
		}

		//remove legals which are not part of our new propnet
		Iterator<Set<Component>> disjointIter = disjointSets.iterator();
		Iterator<Set<Component>> inputIter = disjointInputs.iterator();
		Set<Component> inputGoal = new HashSet<>(goalDependency.keySet());
		while (disjointIter.hasNext()) {
			Set<Component> disjointInput = inputIter.next();
			Set<Component> disjointSet = disjointIter.next();

			if (disjointInput.isEmpty()) {
				inputIter.remove();
				disjointIter.remove();
				continue;
			}

			//add legals that may be missing
			for (Component c : inputGoal) {
				Proposition l = prop.getLegalInputMap().get((Proposition) c);
				if (l == null) {
					continue;
				}
				disjointSet.add(l);
				disjointSet.add(c);
				System.out.println("Adding: " + l.getName());
				//don't forget to add daddy
				disjointSet.addAll(l.getInputs());
				for (Component d : l.getInputs()) {
					disjointSet.addAll(d.getOutputs());
				}
			}

			//remove all legals for inputs not connected to relevant goal
			Iterator<Component> iter = disjointInput.iterator();
			Set<Component> toRemove = new HashSet<>();
			while (iter.hasNext()) {
				Component c = iter.next();
				if (!(c instanceof Proposition)) {
					continue;
				}
				Proposition p = (Proposition) c;
				Proposition l = prop.getLegalInputMap().get(p);
				if (!inputGoal.contains(p)) {
					for (Component in : l.getInputs()) {
						in.removeOutput(l);
					}
					System.out.println("Removing: " + l.getName());
					System.out.println("Detaching: " + p.getName());

					toRemove.add(l);
					//i think we should be doing this
					for (Component in : p.getInputs()) {
						in.removeOutput(p);
					}
					p.removeAllInputs();
					for (Component out : p.getOutputs()) {
						out.removeInput(p);
					}
					p.removeAllOutputs();
				}
			}
			for (Component c : toRemove) {
				disjointSet.remove(c);
			}

			//remove all legals without a matching input in the disjoint set
			for (Component c : disjointSet) {
				if (!(c instanceof Proposition)) {
					continue;
				}
				Proposition p = (Proposition) c;
				if (prop.getInputPropositions().values().contains(p)) {
					continue;
				}
				Proposition li = prop.getLegalInputMap().get(p);
				if (li != null && !disjointSet.contains(li)) {
					for (Component in : p.getInputs()) {
						in.removeOutput(p);
					}
					toRemove.add(p);
					System.out.println("Removing2: " + p.getName() + "  " + li.getName());
				}
			}
			for (Component rem : toRemove) {
				disjointSet.remove(rem);
			}
		}


		Set<Component> trimmedComponents = new HashSet<Component>();
		trimmedComponents.addAll(disjointSets.get(0));

		PropNet prop2 = new PropNet(prop.getRoles(), trimmedComponents);

		//add pseudo NOOPS if needed
		//when is it need one may ask?
		//well, whenever we trim the move set of either role
		//but not our own role
		for (Role rol : prop2.getRoles()) {
			if ( !rol.equals(r) && (prop2.getLegalPropositions().get(rol) == null || prop2.getLegalPropositions().get(rol).size() < prop.getLegalPropositions().get(rol).size()) ) {
				//Set<Proposition> pseudoNoop = new HashSet<>();
				ImmutableList<GdlTerm> body = ImmutableList.of((GdlTerm) rol.getName(), GdlPool.getConstant("noope"));
				Proposition noopLegal = new Proposition(GdlPool.getRelation(GdlPool.getConstant("legal"), body));
				Constant c = new Constant(true);
				c.addOutput(noopLegal);
				noopLegal.addInput(c);
				Proposition noopInput = new Proposition(GdlPool.getRelation(GdlPool.getConstant("does"), body));

				trimmedComponents.add(noopLegal);
				trimmedComponents.add(noopInput);
				trimmedComponents.add(c);
				System.out.println(noopLegal);
				System.out.println(noopInput);
			}
		}
		prop2 = new PropNet(prop.getRoles(), trimmedComponents);



		prop2.renderToFile("factoredprop.dot");

		System.out.println("Original roles: " + prop.getRoles().size());
		System.out.println("Original components: " + prop.getComponents().size());
		System.out.println("Original inputs: " + prop.getInputPropositions().values().size());
		System.out.println("Original legal proposition map: " + prop.getLegalPropositions().size());
		System.out.println("Original legals role: " + prop.getLegalPropositions().get(r).size());


		System.out.println("Factored roles: " + prop2.getRoles().size());
		System.out.println("Factored components: " + prop2.getComponents().size());
		System.out.println("Factored inputs: " + prop2.getInputPropositions().values().size());
		System.out.println("Factored legal proposition map: " + prop2.getLegalPropositions().size());
		System.out.println("Factored legals role: " + prop2.getLegalPropositions().get(r).size());




		//try to factor split on OR terminal
		/*
		Proposition terminal = prop2.getTerminalProposition();
		Set<Component> terminalInputs = terminal.getInputs();
		if (terminalInputs.size() == 1 && terminal.getSingleInput() instanceof Or) { //terminal connected by ORs
			Component terminalOr = terminal.getSingleInput();
			Set<Component> stepCounter = new HashSet<Component>();
			List<Component> orInputs = new ArrayList<Component>();
			Iterator<Component> iter = terminalOr.getInputs().iterator();
			while (iter.hasNext()) {
				Set<Component> sc = new HashSet<Component>();
				Component orInput = iter.next();
				boolean isStepCounter = PropNet.stepCounterDetection(orInput, 0, sc, terminalOr, prop.getInitProposition());
				if (isStepCounter) {
					stepCounter = sc;
				} else {
					orInputs.add(orInput);
				}
			}


		}
		*/



		List<PropNet> factoredProps = new ArrayList<>();
		factoredProps.add(prop2);
		//factoredProps.add(prop3);

		return factoredProps;
	}

	private void inputDFS(Component c, int dist, Map<Component, Integer> goalDependency, Set<Component> seen, PropNet prop, String path) {
		if (seen.contains(c)) {
			return;
		}
		seen.add(c);
		if ((c instanceof Proposition)){
			Proposition p = (Proposition) c;
			boolean isInput = prop.getInputPropositions().values().contains(p);
			if (isInput) {
				goalDependency.put(p, dist);
				//System.out.println(path);
			}
		}
		for (Component in : c.getInputs()) {
			inputDFS(in, dist+1, goalDependency, seen, prop, path+" "+in);
		}
	}

    private int[][] initDependencyMatrix(int length) {

    	int[][] dependencyMatrix = new int[length][length];

    	//let's initialize each entry to -1 for some reason
    	for(int i=0;i<dependencyMatrix.length;++i) {
    		for(int j=0;j<dependencyMatrix.length;++j) {
    			dependencyMatrix[i][j] = -1;
    		}
    	}
    	return dependencyMatrix;
    }

    private static Set<Component> findSet(List<Set<Component>> sets, Component elem) {
    	for (Set<Component> set : sets) {
    		if (set.contains(elem)) {
    			return set;
    		}
    	}
		return null;
    }

	public Pair<PropNet, Integer> removeStepCounter() {
		Proposition terminal = prop.getTerminalProposition();
		Set<Component> terminalInputs = terminal.getInputs();
		if (terminalInputs.size() == 1 && terminal.getSingleInput() instanceof Or) { //terminal connected by ORs
			Component terminalOr = terminal.getSingleInput();
			Iterator<Component> iter = terminalOr.getInputs().iterator();
			while (iter.hasNext()) {
				Set<Component> stepCounter = new HashSet<Component>();
				Component orInput = iter.next();
				boolean isStepCounter = stepCounterDetection(orInput, 0, stepCounter, terminalOr, prop.getInitProposition());
				if (isStepCounter) {
					Set<Component> counterlessComponents = new HashSet<>(prop.getComponents());
					counterlessComponents.removeAll(stepCounter);
					iter.remove();
					PropNet counterlessProp = new PropNet(prop.getRoles(), counterlessComponents);
					return Pair.of(counterlessProp, (stepCounter.size()/3+1));
				}
			}
		} else if (terminalInputs.size() == 1 && terminal.getSingleInput() instanceof Proposition) {//terminal potentially connected by just stepcounter
			System.out.println("Step counter is only input to terminal");
			Set<Component> stepCounter = new HashSet<Component>();
			boolean isStepCounter = stepCounterDetection(terminal.getSingleInput(), 0, stepCounter, null, prop.getInitProposition());
			if (isStepCounter) {
				Set<Component> counterlessComponents = new HashSet<>(prop.getComponents());
				counterlessComponents.removeAll(stepCounter);
				terminal.removeAllInputs();
				PropNet counterlessProp = new PropNet(prop.getRoles(), counterlessComponents);
				return Pair.of(counterlessProp, (stepCounter.size()/3+1));
			}
		}
		return null;
	}

	private boolean stepCounterDetection(Component c, int state, Set<Component> stepCounter, Component last, Proposition init) {
		state = state % 3;
		stepCounter.add(c);
		if (state == 0) {
			if (!(c instanceof Proposition) || c.getInputs().size() != 1) {
				return false;
			}
			boolean isStepCounter = stepCounterDetection(c.getSingleInput(), state+1, stepCounter, c, init);
			if (isStepCounter) {
				Set<Component> toRemove = new HashSet<>();
				for (Component output : c.getOutputs()) {
					if (output.equals(last)) {
						continue;
					}
					toRemove.add(output);
					output.removeInput(c);
				}
				if (toRemove.size() > 0) {
					System.out.println("We have other outputs coming out of the step counter, this can be dangerous ;)");
				}
				for (Component r : toRemove) {
					c.removeOutput(r);
				}
			}
			return isStepCounter;
		}
		if (state == 1) {
			if (!(c instanceof Transition) || c.getInputs().size() != 1) {
				return false;
			}
			return stepCounterDetection(c.getSingleInput(), state+1, stepCounter, c, init);
		}
		if (state == 2) {
			if (c.equals(init)) { //omg we found the step counter, probably
				System.out.println("FOUND STEP COUNTER " + stepCounter.size());
				stepCounter.remove(c);
				c.removeOutput(last);
				return true;
			}
			if (!(c instanceof Proposition) || c.getInputs().size() != 1) {
				return false;
			}
			return stepCounterDetection(c.getSingleInput(), state+1, stepCounter, c, init);
		}
		return false;
	}

}

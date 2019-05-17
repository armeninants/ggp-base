package org.ggp.base.player.gamer.statemachine;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Stack;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import org.apache.lucene.util.OpenBitSet;
import org.ggp.base.apps.player.Player;
import org.ggp.base.player.gamer.exception.GamePreviewException;
import org.ggp.base.util.game.Game;
import org.ggp.base.util.gdl.grammar.GdlPool;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.Role;
import org.ggp.base.util.statemachine.ThreadStateMachine;
import org.ggp.base.util.statemachine.XStateMachine;
import org.ggp.base.util.statemachine.exceptions.GoalDefinitionException;
import org.ggp.base.util.statemachine.exceptions.MoveDefinitionException;
import org.ggp.base.util.statemachine.exceptions.TransitionDefinitionException;

import com.google.common.collect.Lists;

public class MemoryGoat extends XStateMachineGamer {
	public static final boolean SAVE_MATCH = false;
	public static final boolean FACTOR = true;

	protected Player p;
	private XStateMachine machine;
	private List<Role> roles;
	private int self_index, num_threads;
	private volatile int depthCharges, last_depthCharges;
	private long finishBy;
	private long startedAt;
	private int num_roots;
	private volatile XNodeLight[] roots;
	private volatile XNodeLight[] rootsAbsolute;
	private volatile XNodeLight rootSavedAbsolute;
	private volatile Map<OpenBitSet, XNodeLight> savedNodes;
	private List<XNodeLight> path;
	private CompletionService<Struct> executor;
	private ThreadPoolExecutor thread_pool;
	private Thread thread;
	private ThreadStateMachine[] thread_machines;
	private ThreadStateMachine background_machine;
	private ThreadStateMachine solver_machine;
	private volatile int num_charges, num_per;
	private volatile double total_background = 0;
	private volatile double total_threadpool = 0;
	private volatile int loops = 0;
	private volatile int play_loops = 0;

	private HyperParameters hyperparams;
	private volatile double[] C_CONST;
	private volatile double EPSILON = .1;
	private Random rand = new Random();

	protected List<XNodeLight> matchPath;
	protected Map<String, Double> heuristicWeights;


	public class Struct {
		public double[] v;
		public List<XNodeLight> p;
		public int n;

		public Struct(double[] val, List<XNodeLight> arr, int num) {
			this.v = val;
			this.p = arr;
			this.n = num;
		}
	}

	@Override
	public XStateMachine getInitialStateMachine() {
		XStateMachine machine = loadStateMachine();
		if (machine == null) {
			machine = new XStateMachine(FACTOR, new Role(getRoleName()));
		}
		return machine;
	}

	@Override
	public void stateMachineMetaGame(long timeout) throws TransitionDefinitionException, MoveDefinitionException,
			GoalDefinitionException, InterruptedException, ExecutionException {
		initialize(timeout);

		int num_rests = (int) ((finishBy - System.currentTimeMillis()) / 1000) - 1;
		if (num_rests < 0) {
			return;
		}
		for (int i = 0; i < num_rests; ++i) {
			Thread.sleep(1000);
			double avg_back = total_background/loops;
			double avg_threadpool = total_threadpool/play_loops;
			double num = 10 * num_threads * (avg_back/avg_threadpool);
			num_per = (int) num;
			if (num_per < 1) num_per = 1;
			if (System.currentTimeMillis() > finishBy-600) {
				num_rests = i+1;
				break;
			}
		}
		System.out.println("C_CONST: " + hyperparams.C);
		System.out.println("Depth Charges: " + depthCharges);
		System.out.println("Avg Background: " + total_background/loops);
		System.out.println("Avg Threadpool: " + total_threadpool/play_loops);
		System.out.println("Number of playouts per thread: " + num_per);
		last_depthCharges = 0;

		//gather tree shape stats
		List<Double> breadth = new ArrayList<Double>();
		List<Double> nodes_searched = new ArrayList<Double>();
		breadth.add(0.);
		nodes_searched.add(1.);
		double depth = 0.;
		Stack<XNodeLight> q = new Stack<XNodeLight>();
		Stack<Integer> q_d = new Stack<Integer>();
		q.push(roots[num_roots]);
		q_d.push(1);
		System.out.println("current time: " + System.currentTimeMillis());
		System.out.println("time limit: " + finishBy);
		while (!q.isEmpty() && System.currentTimeMillis() < finishBy+400) {
			XNodeLight n = q.pop();
			Integer d = q_d.pop();
			if (nodes_searched.size() < d) {
				nodes_searched.add(1.);
			}
			double cur_ns = nodes_searched.get(d-1);
			nodes_searched.remove(d-1);
			nodes_searched.add(d-1, cur_ns + 1.);
			if (machine.isTerminal(n.state)) {
				depth += d;
				continue;
			}
			if (breadth.size() < d) {
				breadth.add(0.);
			}
			double cur_breadth = breadth.get(d-1);
			breadth.remove(d-1);
			breadth.add(d-1, cur_breadth + machine.getLegalJointMoves(n.state).size());
			//System.out.print(breadth + " ");
			//System.out.print(machine.getLegalJointMoves(n.state).size() + " ");
			Collection<XNodeLight> children = n.getChildren(machine, self_index).values();
			int[] charge_depth = new int[1];
			machine.performDepthCharge(n.state, charge_depth);
			depth += charge_depth[0] + d;
			for (XNodeLight child : children) {
				q.push(child);
				q_d.push(d+1);
			}
			//System.out.println(q_d.peek() + " " + q.peek().toString());
		}
		double size_est = 1.;
		for (int i = 0; i < breadth.size(); ++i) {
			System.out.println(i + ": breadth: " + breadth.get(i) + " searched: " + nodes_searched.get(i));
			size_est *= breadth.get(i) / nodes_searched.get(i);
		}
		double charge_est = depthCharges / (num_rests) * getMatch().getPlayClock();
		//System.out.println("Nodes searched for data: " + nodes_searched);
		//System.out.println("breadth ~ " + avg_breadth);
		//System.out.println("depth ~ " + avg_depth);
		System.out.println("size ~ " + size_est);
		System.out.println("charge ~ " + charge_est);


		thread.suspend();
		//num_roots = (int) (Math.log(size_est) / 2.);
		num_roots = (int) (Math.sqrt(Math.sqrt(charge_est) / Math.log(size_est)));
		num_roots = Math.min(num_roots, 3);
		System.out.println("# roots: " + num_roots);
		initializeRoots();
		thread.resume();
	}

	protected void initialize(long timeout) throws MoveDefinitionException, TransitionDefinitionException, InterruptedException {
		heuristicWeights = new HashMap<String, Double>();
		heuristicWeights.put("differential", 16.);

		matchPath = new ArrayList<XNodeLight>();
		num_roots = 0;

		machine = getStateMachine();
		roles = machine.getRoles();
		self_index = roles.indexOf(getRole());
		background_machine = new ThreadStateMachine(machine,self_index);


		//lets load the tree here
		//rootSavedAbsolute = loadTree();
		savedNodes = loadGraph();
		hyperparams = loadHyperParameters();
		//rootSavedAbsolute = savedNodes.get(getCurrentState());
		System.out.println("Number of loaded nodes: " + savedNodes.size());

		initializeRoots();

		num_charges = 1;
		num_per = Runtime.getRuntime().availableProcessors();
		num_threads = Runtime.getRuntime().availableProcessors();
		thread_pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(num_threads);
		executor = new ExecutorCompletionService<Struct>(thread_pool);
		thread_machines = new ThreadStateMachine[num_threads];
		for (int i = 0; i < num_threads; ++i) {
			thread_machines[i] = new ThreadStateMachine(machine,self_index);
		}


		thread = new Thread(new runMCTS());
		depthCharges = 0;
		last_depthCharges = 0;
		thread.start();

		finishBy = timeout - 2500;
		System.out.println("NumThreads: " + num_threads);
	}

	protected void initializeRoots() throws MoveDefinitionException, TransitionDefinitionException {
		rootsAbsolute = new XNodeLight[num_roots+1];
		roots = new XNodeLight[num_roots+1];
		C_CONST = new double[num_roots+1];
		for (int i = 0; i < num_roots; ++i) {
			rootsAbsolute[i] = generateXNode(getCurrentState(), roles.size(), i);
			roots[i] = rootsAbsolute[i];
			Expand(roots[i], null, i);
			C_CONST[i] = HyperParameters.generateC(hyperparams.C, hyperparams.C/4);
		}

		rootSavedAbsolute = generateXNode(getCurrentState(), roles.size(), num_roots);
		Expand(rootSavedAbsolute, null, num_roots);

		rootsAbsolute[rootsAbsolute.length-1] = rootSavedAbsolute;
		roots[roots.length-1] = rootSavedAbsolute;
		C_CONST[roots.length-1] = HyperParameters.generateC(hyperparams.C, hyperparams.C/4);
		System.out.println("Initialized " + (num_roots+1) + " roots");
	}

	@Override
	public Move stateMachineSelectMove(long timeout)
			throws TransitionDefinitionException, MoveDefinitionException,
			GoalDefinitionException, InterruptedException, ExecutionException {
		//More efficient to use Compulsive Deliberation for one player games
		//Use two-player implementation for two player games
		depthCharges = 0;
		total_background = 0;
		total_threadpool = 0;
		loops = 0;
		play_loops = 0;
		System.out.println("Background Depth Charges: " + last_depthCharges);
		finishBy = timeout - 2500;
		startedAt = System.currentTimeMillis();

		return MCTS();
	}

	protected void initializeMCTS() throws MoveDefinitionException, TransitionDefinitionException, InterruptedException {
		OpenBitSet currentState = getCurrentState();

		for (int i = 0; i < roots.length; ++i) {
			if (roots[i] == null) System.out.println("NULL ROOT");
			if (roots[i].state.equals(currentState)) continue;
			boolean found_next = false;
			for (List<Move> jointMove : machine.getLegalJointMoves(roots[i].state)) {
				OpenBitSet nextState = machine.getNextState(roots[i].state, jointMove);
				if (currentState.equals(nextState)) {
					roots[i] = roots[i].getChildren(machine, self_index).get(jointMove);
					if (roots[i] == null) {
						System.out.println("NOT IN MAP");
						break;
					}
					found_next = true;
					break;
				}
			}
			if (!found_next) {
				System.out.println("ERROR. Current State not in tree");
				roots[i] = generateXNode(currentState, roles.size(), i);
			}
		}

		matchPath.add(roots[roots.length-1]);

		//restart the worst root
		if (roots.length > 1) {
			int worstRoot = 0;
			double worstScore = 100.1;
			for (int i=0; i < num_roots; ++i) {
				double utility = roots[i].utility[self_index] / roots[i].updates;
				if (utility < worstScore) {
					worstRoot = i;
					worstScore = utility;
				}
			}
			System.out.println("Restarting Root at index " + worstRoot);
			roots[worstRoot] = generateXNode(currentState, roles.size(), worstRoot);
			C_CONST[worstRoot] = HyperParameters.generateC(hyperparams.C, hyperparams.C/4);
		}
	}

	protected Move MCTS() throws MoveDefinitionException, TransitionDefinitionException, GoalDefinitionException, InterruptedException, ExecutionException {
		initializeMCTS();
		thread_pool.getQueue().clear();
		//graph.clear();
		int num_rests = (int) ((finishBy - System.currentTimeMillis()) / 1000);
		if (num_rests < 0) {
			return bestMove();
		}
		for (int i = 0; i < num_rests; ++i) {
			Thread.sleep(1000);
			double avg_back = total_background/loops;
			double avg_threadpool = total_threadpool/play_loops;
			double num = 10 * num_threads * (avg_back/avg_threadpool);
			num_per = (int) num;
			if (num_per < 1) num_per = 1;
			if (System.currentTimeMillis() > finishBy-400) {
				num_rests = i+1;
				break;
			}
		}
		System.out.println("C_CONST: " + hyperparams.C);
		System.out.println("Depth Charges: " + depthCharges);
		System.out.println("Number of Select/Expand Loops " + loops);
		/*System.out.println("Avg Select: " + total_select/loops);
		System.out.println("Avg Expand: " + total_expand/loops);
		System.out.println("Avg Backprop: " + total_backpropagate/depthCharges);
		System.out.println("Avg Playout: " + total_playout/play_loops);*/
		System.out.println("Avg Background: " + total_background/loops);
		System.out.println("Avg Threadpool: " + total_threadpool/play_loops);
		System.out.println("Number of playouts per thread: " + num_per);
		last_depthCharges = 0;
		return bestMove();
	}

	public class solver implements Runnable {
		@Override
		public void run() {

		}
	}

	public class runMCTS implements Runnable {
		@Override
		public void run() {
			XNodeLight root_thread;
			while (true) {
				double start = System.currentTimeMillis();
				int rand_idx = rand.nextInt(roots.length);
				root_thread = roots[rand_idx];
				path = new ArrayList<XNodeLight>();
				/*if (rand_idx == num_roots && !matchPath.isEmpty()) {
					path.addAll(matchPath);
				} else {
					path.add(root_thread);
				}*/
				path.add(root_thread);
				//double select_start = System.currentTimeMillis();
				try {
					Select(root_thread, path, rand_idx);
				} catch (MoveDefinitionException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				//total_select += (System.currentTimeMillis() - select_start);
				XNodeLight n = path.get(path.size() - 1);
				//double expand_start = System.currentTimeMillis();
				try {
					Expand(n, path, rand_idx);
				} catch (MoveDefinitionException | TransitionDefinitionException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				n = path.get(path.size() - 1);
				//total_expand += (System.currentTimeMillis() - expand_start);
				// spawn off multiple threads
				executor.submit(new RunMe(n, path, num_per));
				//executor.submit(new RunMe(path.get(0), path, num_per));


				while(true) {
					Future<Struct> f = executor.poll();

					if (f == null) break;
					Struct s = null;
			        try {
						s = f.get();
					} catch (InterruptedException | ExecutionException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
			        //double back_start = System.currentTimeMillis();
			        int num = s.n;
			        Backpropogate(s.v,s.p, num);
			        //total_backpropagate += (System.currentTimeMillis() - back_start);
			        depthCharges += num;
			        last_depthCharges += num;
			    }
				total_background += (System.currentTimeMillis() - start);
				++loops;
			}
		}
	}

	public class RunMe implements Callable<Struct> {
		private XNodeLight node;
		private List<XNodeLight> p;
		private int num;

		public RunMe(XNodeLight n, List<XNodeLight> arr, int number) {
			this.node = n;
			this.p = arr;
			this.num = number;
		}
		@Override
		public Struct call() throws InterruptedException{
			double start = System.currentTimeMillis();
			double[] val = new double[roles.size()];
			double[] curr = null;
			int thread_ind = (int) (Thread.currentThread().getId() % num_threads);
			ThreadStateMachine mac = thread_machines[thread_ind];
			for (int i = 0; i < num; ++i) {
				//double start = System.currentTimeMillis();
				try {
					curr = mac.Playout(node);
				} catch (MoveDefinitionException | TransitionDefinitionException | GoalDefinitionException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				for (int j = 0; j < roles.size(); ++j) {
					val[j] += curr[j];
				}
				//++play_loops;
				//total_playout += (System.currentTimeMillis() - start);
			}
			++play_loops;
			total_threadpool += (System.currentTimeMillis() - start);
			Struct s = new Struct(val, p, num);
			return s;
	    }
	}

	protected Move bestMove() throws MoveDefinitionException {
		double maxValue = Double.NEGATIVE_INFINITY;
		Move maxMove = roots[roots.length-1].getLegalMoves(machine, self_index)[0];
		int maxIdx = num_roots;
		Move maxSavedMove = roots[roots.length-1].getLegalMoves(machine, self_index)[0];

		int root_idx = 0;
		for (XNodeLight n : roots) {
			System.out.println("-- C: " + C_CONST[root_idx]);
			int size = n.getLegalMoves(machine, self_index).length;
			for(int i = 0; i < size; ++i) {
				Move move = n.getLegalMoves(machine, self_index)[i];
				double oppValue = Double.NEGATIVE_INFINITY;
				double myValue = Double.NEGATIVE_INFINITY;
				double visits = 0;
				for (List<Move> jointMove : n.getLegalJointMoves(machine, self_index).get(move)) {
					XNodeLight succNode = n.getChildren(machine, self_index).get(jointMove);
					if (succNode.updates != 0) {
						//double nodeValue = succNode.utility[self_index] / succNode.updates;
						double nodeValue = averageOpponentUtility(succNode);
						if (nodeValue > oppValue) {
							visits = succNode.updates;
							oppValue = nodeValue;
							myValue = succNode.utility[self_index] / succNode.updates;
						}
					}
					if (myValue == Double.NEGATIVE_INFINITY) System.out.println("visits: " + succNode.visits + " updates: " + succNode.updates + " utility: " + succNode.utility.toString());
				}
				System.out.println("Move: " + move + " Value: " + (myValue == Double.NEGATIVE_INFINITY ? "N/A" : String.valueOf(myValue)) + " Visits: " + visits);
				if (myValue > maxValue && myValue != Double.NEGATIVE_INFINITY) {
					maxValue = myValue;
					maxMove = move;
					maxIdx = root_idx;
					if (root_idx == num_roots) {
						maxSavedMove = move;
					}
				}
			}
			++root_idx;
		}
		System.out.println("--");
		System.out.println(getName() + " Max Move: " + maxMove + " Max Value: " + maxValue + " Root Index: " + maxIdx);
		if (!maxMove.equals(maxSavedMove)) {
			hyperparams.C = hyperparams.C * .95 + C_CONST[maxIdx] * .05;
		}

		return maxMove;
	}

	protected void Backpropogate(double[] val, List<XNodeLight> path, int num) {
		int size = path.size();
		XNodeLight nod = path.get(size - 1);
		for (int i = 0; i < size; ++i) {
			nod = path.get(i);
			for (int j = 0; j < roles.size(); ++j) {
				nod.utility[j] += val[j];
			}
			nod.updates += num;
			nod.visits++;
		}
	}

	protected void Select(XNodeLight n, List<XNodeLight> path, int root_idx) throws MoveDefinitionException {
		while(true) {
			/*System.out.println(roots.toString());
			System.out.println(n);
			System.out.println(n.children);*/
			//++n.visits;
			if (background_machine.isTerminal(n.state)) return;
			if (n.getChildren(machine, self_index).isEmpty()) return;
			if (n.visits == 0) return;
			double maxValue = Double.NEGATIVE_INFINITY;
			double parentVal = C_CONST[root_idx] * Math.sqrt(Math.log(n.updates));
			XNodeLight maxChild = null;

			int size = n.getLegalMoves(machine, self_index).length;

			//select a random move with decaying probability
			/*List<List<Move>> jointMoves = n.getLegalJointMoves(machine, self_index)
										   .get(n.getLegalMoves(machine, self_index)[rand.nextInt(size)]);
			EPSILON = .00 * (finishBy - System.currentTimeMillis()) / (finishBy - startedAt);
			if (rand.nextDouble() < EPSILON) {
				//System.out.println("RANDOM EPSILON SELECT");
				XNodeLight succNode = n.getChildren(machine, self_index).get(jointMoves.get(rand.nextInt(jointMoves.size())));
				if (succNode.visits == 0) {
					++succNode.visits;
					path.add(succNode);
					return;
				}
				maxChild = succNode;
				path.add(maxChild);
				n = maxChild;
				continue;
			}*/

			for(int i = 0; i < size; ++i) {
				Move move = n.getLegalMoves(machine, self_index)[i];

				double minValue = Double.NEGATIVE_INFINITY;
				XNodeLight minChild = null;
				for (List<Move> jointMove : n.getLegalJointMoves(machine, self_index).get(move)) {
					XNodeLight succNode = n.getChildren(machine, self_index).get(jointMove);
					if (succNode.visits == 0) {
						//++succNode.visits;
						path.add(succNode);

						//Double diff = differentialUtility(succNode, n);
						//System.out.println(move + " " + diff);

						//get to furthest depth expanded, randomly
						while (!succNode.getChildren(machine, self_index).isEmpty()) {
							int numChildren = succNode.getChildren(machine, self_index).size();
							List<XNodeLight> succChildren = Lists.newArrayList(succNode.getChildren(machine, self_index).values());
							XNodeLight randChild = succChildren.get(rand.nextInt(numChildren));
							path.add(randChild);
							succNode = randChild;
						}
						return;
					}
					double nodeValue = uctMin(succNode, parentVal);
					//System.out.println("min: " + jointMove.get(self_index) + " " + nodeValue);
					if (nodeValue > minValue) {
						minValue = nodeValue;
						minChild = succNode;
					}
				}
				minValue = uctMax(minChild, parentVal);
				//System.out.println("max: " + minValue);
				if (minValue > maxValue) {
					maxValue = minValue;
					maxChild = minChild;
				}
			}

			path.add(maxChild);
			n = maxChild;
		}
	}


	protected double uctMin(XNodeLight n, double parentVisits) {
		double value = averageOpponentUtility(n);
		return value + (parentVisits / Math.sqrt(n.updates));
	}

	protected double uctMax(XNodeLight n, double parentVisits) {
		//System.out.println(n);
		double value = n.utility[self_index] / 100. / n.updates;
		return value + (parentVisits / Math.sqrt(n.updates));
	}

	protected double averageOpponentUtility(XNodeLight n) {
		double value = 0.;
		if (roles.size() > 1) {
			double utility = 0;
			for (int i = 0; i < roles.size(); ++i) {
				if (i == self_index) continue;
				utility += n.utility[i];
			}
			value = utility / 100. / (roles.size() - 1) / n.updates;
		}
		return value;
	}

	protected double differentialUtility(XNodeLight n, XNodeLight parent) {
		Long xorCount = OpenBitSet.xorCount(n.state, parent.state);
		//return xorCount.doubleValue();
		return xorCount.doubleValue() / n.state.size() * 100.;
	}

	protected void Expand(XNodeLight n, List<XNodeLight> path, int root_idx) throws MoveDefinitionException, TransitionDefinitionException {
		if (n.getChildren(machine, self_index).isEmpty() && !background_machine.isTerminal(n.state)) {
			List<Move> moves = background_machine.getLegalMoves(n.state, self_index);
			int size = moves.size();
			n.setLegalMoves(moves.toArray(new Move[size]));
			for (int i = 0; i < size; ++i) {
				Move move = n.getLegalMoves(machine, self_index)[i];
				n.getLegalJointMoves(machine, self_index).put(move, new ArrayList<List<Move>>());
			}
			for (List<Move> jointMove: background_machine.getLegalJointMoves(n.state)) {
				OpenBitSet state = background_machine.getNextState(n.state, jointMove);
				XNodeLight child = n.getChildren(machine, self_index).get(jointMove);
				if(child == null) {
					child = generateXNode(state, roles.size(), root_idx);

					child.utility[self_index] += differentialUtility(child, n) * heuristicWeights.get("differential");
					child.visits += heuristicWeights.get("differential");
					child.updates += heuristicWeights.get("differential");

					n.getLegalJointMoves(machine, self_index).get(jointMove.get(self_index)).add(jointMove);
					n.getChildren(machine, self_index).put(jointMove, child);
				}

			}
			if (path != null) {
				path.add(n.getChildren(machine, self_index).get(background_machine.getRandomJointMove(n.state)));
			}
		} else if (!background_machine.isTerminal(n.state)) {
			//System.out.println("ERROR. Tried to expand node that was previously expanded (2)");
		}
	}

	/*protected void Expand(XNodeLight n, int root_idx) throws MoveDefinitionException, TransitionDefinitionException {//Assume only expand from max node

		if (n.getChildren(machine, self_index).isEmpty() && !machine.isTerminal(n.state)) {
			List<Move> moves = machine.getLegalMoves(n.state, self_index);
			int size = moves.size();
			n.setLegalMoves(moves.toArray(new Move[size]));
			for (int i = 0; i < size; ++i) {
				Move move = n.getLegalMoves(machine, self_index)[i];
				n.getLegalJointMoves(machine, self_index).put(move, new ArrayList<List<Move>>());
			}
			for (List<Move> jointMove: machine.getLegalJointMoves(n.state)) {
				OpenBitSet state = machine.getNextState(n.state, jointMove);
				XNodeLight child = n.getChildren(machine, self_index).get(jointMove);
				if(child == null) {
					child = generateXNode(state, roles.size(), root_idx);
					n.getLegalJointMoves(machine, self_index).get(jointMove.get(self_index)).add(jointMove);
					n.getChildren(machine, self_index).put(jointMove, child);
				}

			}
		} else if (!machine.isTerminal(n.state)) {
			System.out.println("ERROR. Tried to expand node that was previously expanded (1)");
		}
	}*/

	@Override
	public void stateMachineStop() {
		try {
			pruneTree(rootSavedAbsolute);
		} catch (MoveDefinitionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//saveTree();
		if (SAVE_MATCH) {
			saveGraph();
			saveMachine();
			saveHyperParameters();
		}
		cleanup();
	}

	@Override
	public void stateMachineAbort() {
		cleanup();
	}

	protected void cleanup() {

		GdlPool.drainPool();
		thread_pool.shutdownNow();
		thread.stop();
	}

	protected void saveTree() {
		int numRules = getMatch().getGame().getRules().size();
		String filename = "gametree" + numRules + ".goat";
		try {
            //Saving of object in a file
            FileOutputStream file = new FileOutputStream(filename);
            ObjectOutputStream out = new ObjectOutputStream(file);

            // Method for serialization of object
            //out.writeObject(rootAbsolute);
            out.writeObject(rootSavedAbsolute);

            out.close();
            file.close();

            System.out.println("Saved game tree in " + filename);

        } catch(IOException ex) {
            System.out.println("IOException when writing to file " + filename);
        }
	}

	protected XNodeLight loadTree() {
		int numRules = getMatch().getGame().getRules().size();
		String filename = "gametree" + numRules + ".goat";
		// Deserialization
		XNodeLight savedRoot = null;
        try
        {
            // Reading the object from a file
            FileInputStream file = new FileInputStream(filename);
            ObjectInputStream in = new ObjectInputStream(file);

            // Method for deserialization of object
            savedRoot = (XNodeLight) in.readObject();

            in.close();
            file.close();

            System.out.println("Loaded game tree from " + filename);
        }

        catch(IOException ex)
        {
            System.out.println("IOException when loading from file " + filename);
        }

        catch(ClassNotFoundException ex)
        {
            System.out.println("ClassNotFoundException when loading from file " + filename);
        }
        return savedRoot;
	}

	protected void saveGraph() {
		int numRules = getMatch().getGame().getRules().size();
		String filename = "gamegraph" + numRules + ".goat";
		try {
            //Saving of object in a file
            FileOutputStream file = new FileOutputStream(filename);
            ObjectOutputStream out = new ObjectOutputStream(file);

            // Method for serialization of object
            //out.writeObject(rootAbsolute);
            out.writeObject(savedNodes);

            out.close();
            file.close();

            System.out.println("Saved game graph in " + filename);

        } catch(IOException ex) {
            System.out.println("IOException when writing to file " + filename);
        }
	}

	protected void saveHyperParameters() {
		int numRules = getMatch().getGame().getRules().size();
		String filename = "gamehyperparams" + numRules + ".goat";
		try {
            //Saving of object in a file
            FileOutputStream file = new FileOutputStream(filename);
            ObjectOutputStream out = new ObjectOutputStream(file);

            // Method for serialization of object
            //out.writeObject(rootAbsolute);
            out.writeObject(hyperparams);

            out.close();
            file.close();

            System.out.println("Saved hyper parameters in " + filename);

        } catch(IOException ex) {
            System.out.println("IOException when writing to file " + filename);
        }
	}

	//construct a new XNode object, while maintaining the graph mapping structure
	protected XNodeLight generateXNode(OpenBitSet state, int numRoles, int rootIdx) {
		XNodeLight node = null;
		XNodeLight savedNode = savedNodes.get(state);
		if (rootIdx == num_roots) {
			node = savedNode;
		}


		if (node == null) {

			//decay our node visits based on saved value
			/*if (savedNode != null) {
				//double visits = Math.log(savedNode.visits) * C_CONST[num_roots];
				//double updates = Math.log(savedNode.updates) * C_CONST[num_roots];
				double[] utility = new double[savedNode.utility.length];
				for (int ui = 0; ui < savedNode.utility.length; ui++) {
					utility[ui] = savedNode.utility[ui] / savedNode.updates;
				}
				node = new XNodeLight(state, 0., 0, utility);
			} else {
				node = new XNodeLight(state, numRoles);
			}*/
			node = new XNodeLight(state, numRoles);

			/*if (background_machine.isTerminal(state)) {
				node.solved = true;
				node.visits = 1;
				try {
					node.utility = background_machine.getCurrGoal(state);
				} catch (GoalDefinitionException e) {
					System.out.println("No goal defined while generating terminal XNode");
				}
			}*/


			if (rootIdx == num_roots) {
				savedNodes.put(state, node);
			}
		}
		/*Set<XNodeLight> stateNodes = graph.get(state);
		if (stateNodes == null) {
			stateNodes = new HashSet<XNodeLight>();
		}
		stateNodes.add(node);*/

		return node;
	}

	@SuppressWarnings("unchecked")
	protected Map<OpenBitSet, XNodeLight> loadGraph() {
		int numRules = getMatch().getGame().getRules().size();
		String filename = "gamegraph" + numRules + ".goat";
		// Deserialization
		Map<OpenBitSet, XNodeLight> savedGraph = null;
        try
        {
            // Reading the object from a file
            FileInputStream file = new FileInputStream(filename);
            ObjectInputStream in = new ObjectInputStream(file);

            // Method for deserialization of object
            savedGraph = (Map<OpenBitSet, XNodeLight>) in.readObject();

            in.close();
            file.close();

            System.out.println("Loaded game graph from " + filename);
        }

        catch(IOException ex)
        {
            System.out.println("IOException when loading from file " + filename);
        }

        catch(ClassNotFoundException ex)
        {
            System.out.println("ClassNotFoundException when loading from file " + filename);
        }

        if (savedGraph == null) {
        	savedGraph = new HashMap<OpenBitSet, XNodeLight>();
        }

        return savedGraph;
	}

	@SuppressWarnings("unchecked")
	protected HyperParameters loadHyperParameters() {
		int numRules = getMatch().getGame().getRules().size();
		String filename = "gamehyperparams" + numRules + ".goat";
		// Deserialization
		HyperParameters savedParams = null;
        try
        {
            // Reading the object from a file
            FileInputStream file = new FileInputStream(filename);
            ObjectInputStream in = new ObjectInputStream(file);

            // Method for deserialization of object
            savedParams = (HyperParameters) in.readObject();

            in.close();
            file.close();

            System.out.println("Loaded hyper parameters from " + filename);
        }

        catch(IOException ex)
        {
            System.out.println("IOException when loading from file " + filename);
        }

        catch(ClassNotFoundException ex)
        {
            System.out.println("ClassNotFoundException when loading from file " + filename);
        }

        if (savedParams == null) {
        	savedParams = new HyperParameters(HyperParameters.generateC(10, 2));
        }

        return savedParams;
	}

	protected void pruneNodes() {
		for (Entry<OpenBitSet, XNodeLight> n : savedNodes.entrySet()) {

		}
	}

	protected void pruneTree(XNodeLight n) throws MoveDefinitionException {
		double p = .1;
		while (n != null && n.getChildren(machine, self_index) != null && !n.getChildren(machine, self_index).isEmpty()) {
			int numChildren = n.getChildren(machine, self_index).size();
			//dont prune and move on to child
			if (rand.nextDouble() >= p || n.getChildren(machine, self_index).size() == 1) {
				int i = 0;
				int r = rand.nextInt(numChildren);
				for (XNodeLight child : n.getChildren(machine, self_index).values()) {
					if (i++ == r) n = child;
				}
			} else { //prune a branch that is the worst
				double worstScore = Double.MAX_VALUE;
				List<Move> worstChild = null;
				for (Entry<List<Move>, XNodeLight> e : n.getChildren(machine, self_index).entrySet()) {
					List<Move> m = e.getKey();
					XNodeLight child = e.getValue();
					double maxNodeScore = -1;
					//get the best utility amongst players
					for (double utility : child.utility) {
						if (utility > maxNodeScore) maxNodeScore = utility;
					}
					if (maxNodeScore < worstScore) {
						worstScore = maxNodeScore;
						worstChild = m;
					}
				}
				n.getChildren(machine, self_index).remove(worstChild);
			}
		}
	}

	@Override
	public void preview(Game g, long timeout) throws GamePreviewException {
		// TODO Auto-generated method stub

	}

	@Override
	public String getName() {
		return "The Goat Remembers";
	}



}

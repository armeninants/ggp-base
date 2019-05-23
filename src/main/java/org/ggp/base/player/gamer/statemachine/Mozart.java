package org.ggp.base.player.gamer.statemachine;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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

public class Mozart extends XStateMachineGamer {
	public static final boolean FACTOR = true;
	public static final boolean CACHING = true;

	protected Player p;
	private XStateMachine machine;
	private List<Role> roles;
	private int self_index, num_threads;
	private volatile int depthCharges, last_depthCharges;
	private long finishBy;
	private long startedAt;
	private int num_roots;
	private volatile XNodeLight[] roots;
	private volatile List<Map<OpenBitSet, XNodeLight>> savedNodes = new ArrayList<Map<OpenBitSet, XNodeLight>>();
	private List<XNodeLight> path;
	private CompletionService<Struct> executor;
	private ThreadPoolExecutor thread_pool;
	private Thread thread;
	private ThreadStateMachine[] thread_machines;
	private ThreadStateMachine background_machine;
	//private ThreadStateMachine solver_machine;
	private volatile int num_charges, num_per;
	private volatile double total_background = 0;
	private volatile double total_threadpool = 0;
	private volatile int loops = 0;
	private volatile int play_loops = 0;

	private HyperParameters hyperparams = new HyperParameters(8.0);
	private volatile double[] C_CONST;
	private Random rand = new Random();

	private volatile boolean thread_stop = false;
	private volatile boolean mcts_thread_running = false;

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
		XStateMachine mach = new XStateMachine(FACTOR, new Role(getRoleName()));
		return mach;
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
			Collection<XNodeLight> children = n.getChildren(self_index).values();
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
		if (num_rests > 0) {
		double charge_est = depthCharges / (num_rests) * getMatch().getPlayClock();
			//System.out.println("Nodes searched for data: " + nodes_searched);
			//System.out.println("breadth ~ " + avg_breadth);
			//System.out.println("depth ~ " + avg_depth);
			System.out.println("size ~ " + size_est);
			System.out.println("charge ~ " + charge_est);


			//num_roots = (int) (Math.log(size_est) / 2.);
			num_roots = (int) (Math.sqrt(Math.sqrt(charge_est) / (Math.log(size_est)) ) / 3.0);
			num_roots = Math.min(num_roots, 3);
		} else {
			num_roots = 0;
		}
		System.out.println("# roots: " + num_roots);
		//thread.suspend();

		//wait for mcts to finish current iteration
		thread_stop = true;
		while (mcts_thread_running) {}

		initializeRoots(false);

		//reinitialize mcts thread
		thread = new Thread(new runMCTS());
		thread.start();

		//thread.resume();
		num_roots = 0;
	}

	protected void initialize(long timeout) throws MoveDefinitionException, TransitionDefinitionException, InterruptedException {
		heuristicWeights = new HashMap<String, Double>();
		heuristicWeights.put("differential", 16.);

		num_roots = 0;

		machine = getStateMachine();
		roles = machine.getRoles();
		self_index = roles.indexOf(getRole());
		System.out.println("Role index: " + self_index);
		background_machine = new ThreadStateMachine(machine,self_index);

		initializeRoots(true);

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

	protected void initializeRoots(Boolean first) throws MoveDefinitionException, TransitionDefinitionException {
		XNodeLight newRoot;
		Map<OpenBitSet, XNodeLight> rootSaved;
		if (!first) {
			newRoot = roots[roots.length-1];
			rootSaved = savedNodes.get(savedNodes.size()-1);
		} else {
			rootSaved = new HashMap<OpenBitSet, XNodeLight>();
			newRoot = generateXNode(getCurrentState(), roles.size(), num_roots, false);
		}

		roots = new XNodeLight[num_roots+1];
		C_CONST = new double[num_roots+1];
		savedNodes = new ArrayList<Map<OpenBitSet, XNodeLight>>();
		for (int i = 0; i < num_roots; ++i) {
			savedNodes.add(new HashMap<OpenBitSet, XNodeLight>());
			roots[i] = generateXNode(getCurrentState(), roles.size(), i, CACHING);
			C_CONST[i] = HyperParameters.generateC(hyperparams.C, hyperparams.C/4);
			Expand(roots[i], null, i);
		}


		roots[roots.length-1] = newRoot;
		savedNodes.add(rootSaved);
		if (first) {
			Expand(newRoot, null, num_roots);
		}

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
		finishBy = timeout - 2700;
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
					roots[i] = roots[i].getChildren(self_index).get(jointMove);
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
				roots[i] = generateXNode(currentState, roles.size(), i, CACHING);
			}
		}

		//restart the worst root
		if (roots.length > 1) {
			int worstRoot = 0;
			int bestRoot = 0;
			double worstScore = 100.1;
			double bestScore = -1.0;
			for (int i=0; i < num_roots; ++i) {
				double utility = roots[i].utility[self_index] / roots[i].updates;
				System.out.println("root: " + i + " utility: " + utility);
				if (utility < worstScore) {
					worstRoot = i;
					worstScore = utility;
				}
				if (utility > bestScore) {
					bestRoot = i;
					bestScore = utility;
				}
			}
			System.out.println("Restarting Root at index " + worstRoot);
			savedNodes.set(worstRoot, new HashMap<OpenBitSet, XNodeLight>());
			roots[worstRoot] = generateXNode(currentState, roles.size(), worstRoot, CACHING);
			C_CONST[worstRoot] = HyperParameters.generateC(C_CONST[bestRoot] - C_CONST[bestRoot]/16, C_CONST[bestRoot]/4);
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
			mcts_thread_running = true;
			XNodeLight root_thread;
			while (!thread_stop) {
				double start = System.currentTimeMillis();
				int rand_idx = rand.nextInt(roots.length);
				root_thread = roots[rand_idx];
				path = new ArrayList<XNodeLight>();

				path.add(root_thread);
				//double select_start = System.currentTimeMillis();
				try {
					Select(root_thread, path, rand_idx);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					System.out.println(path.size());
					System.out.println(rand_idx);
					run();
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
			thread_stop = false;
			mcts_thread_running = false;
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
		Move maxMove = roots[roots.length-1].getLegalMoves(self_index)[0];
		int maxIdx = num_roots;
		Move maxSavedMove = roots[roots.length-1].getLegalMoves(self_index)[0];

		int root_idx = 0;
		for (XNodeLight n : roots) {
			System.out.println("-- C: " + C_CONST[root_idx]);
			int size = n.getLegalMoves(self_index).length;
			for(int i = 0; i < size; ++i) {
				Move move = n.getLegalMoves(self_index)[i];
				double oppValue = Double.NEGATIVE_INFINITY;
				double myValue = Double.NEGATIVE_INFINITY;
				double visits = 0;
				for (List<Move> jointMove : n.getLegalJointMoves(self_index).get(move)) {
					XNodeLight succNode = n.getChildren(self_index).get(jointMove);
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
			if (n.visits == 0) return;
			if (n.getChildren(self_index).isEmpty()) return;
			if (background_machine.isTerminal(n.state)) return;
			double maxValue = Double.NEGATIVE_INFINITY;
			double parentVal = C_CONST[root_idx] * Math.sqrt(Math.log(n.updates));
			XNodeLight maxChild = null;

			int size = n.getLegalMoves(self_index).length;

			for(int i = 0; i < size; ++i) {
				Move move = n.getLegalMoves(self_index)[i];

				double minValue = Double.NEGATIVE_INFINITY;
				XNodeLight minChild = null;
				for (List<Move> jointMove : n.getLegalJointMoves(self_index).get(move)) {
					XNodeLight succNode = n.getChildren(self_index).get(jointMove);
					if (succNode.visits == 0) {
						//++succNode.visits;
						path.add(succNode);

						//Double diff = differentialUtility(succNode, n);
						//System.out.println(move + " " + diff);

						//get to furthest depth expanded, randomly
						while (!succNode.getChildren(self_index).isEmpty()) {
							int numChildren = succNode.getChildren(self_index).size();
							List<XNodeLight> succChildren = Lists.newArrayList(succNode.getChildren(self_index).values());
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
		if (n.getChildren(self_index).isEmpty() && !background_machine.isTerminal(n.state)) {
			List<Move> moves = background_machine.getLegalMoves(n.state, self_index);
			int size = moves.size();
			n.setLegalMoves(moves.toArray(new Move[size]));
			for (int i = 0; i < size; ++i) {
				Move move = n.getLegalMoves(self_index)[i];
				n.getLegalJointMoves(self_index).put(move, new ArrayList<List<Move>>());
			}
			for (List<Move> jointMove: background_machine.getLegalJointMoves(n.state)) {
				OpenBitSet state = background_machine.getNextState(n.state, jointMove);
				XNodeLight child = n.getChildren(self_index).get(jointMove);
				if(child == null) {
					child = generateXNode(state, roles.size(), root_idx, CACHING);

					child.utility[self_index] += differentialUtility(child, n) * heuristicWeights.get("differential");
					child.visits += heuristicWeights.get("differential");
					child.updates += heuristicWeights.get("differential");

					n.getLegalJointMoves(self_index).get(jointMove.get(self_index)).add(jointMove);
					n.getChildren(self_index).put(jointMove, child);
				}

			}
			if (path != null) {
				path.add(n.getChildren(self_index).get(background_machine.getRandomJointMove(n.state)));
			}
		} else if (!background_machine.isTerminal(n.state)) {
			//System.out.println("ERROR. Tried to expand node that was previously expanded (2)");
		}
	}

	@Override
	public void stateMachineStop() {
		cleanup();
	}

	@Override
	public void stateMachineAbort() {
		cleanup();
	}

	protected void cleanup() {

		thread_stop = true;
		GdlPool.drainPool();
		thread_pool.shutdownNow();
		savedNodes = new ArrayList<Map<OpenBitSet, XNodeLight>>();
	}

	//construct a new XNode object, while maintaining the graph mapping structure
	protected XNodeLight generateXNode(OpenBitSet state, int numRoles, int rootIdx, boolean useCache) {
		XNodeLight node = null;
		Map<OpenBitSet, XNodeLight> cache = null;

		if (useCache) {
			cache = savedNodes.get(rootIdx);
			if (cache != null) {
				node = cache.get(state);
			}
		}

		if (node == null) {
			node = new XNodeLight(state, numRoles);
			if (cache != null) {
				cache.put(state, node);
			}
		}

		return node;
	}

	@Override
	public void preview(Game g, long timeout) throws GamePreviewException {
		// TODO Auto-generated method stub

	}

	@Override
	public String getName() {
		return "Mozart the Goat";
	}



}
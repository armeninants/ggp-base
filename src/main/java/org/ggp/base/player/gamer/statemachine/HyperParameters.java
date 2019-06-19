package org.ggp.base.player.gamer.statemachine;

import java.io.Serializable;
import java.util.Random;

public class HyperParameters implements Serializable {

	private static final long serialVersionUID = -5974785139649528310L;
	private static Random r = new Random();

	public double C;

	public HyperParameters(double c) {
		C = c;
	}

	public static double generateC(double mean, double std) {
		return Math.max(1.414, r.nextGaussian()*std + mean);
	}

	@Override
	public String toString() {
		StringBuilder s = new StringBuilder();
		s.append("C: " + C);

		return s.toString();
	}

}

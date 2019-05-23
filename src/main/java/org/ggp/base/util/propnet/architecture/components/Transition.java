package org.ggp.base.util.propnet.architecture.components;

import org.ggp.base.util.propnet.architecture.Component;

/**
 * The Transition class is designed to represent pass-through gates.
 */
@SuppressWarnings("serial")
public final class Transition extends Component
{
	/**
	 * Returns the value of the input to the transition.
	 *
	 * @see org.ggp.base.util.propnet.architecture.Component#getValue()
	 */
	@Override
	public boolean getValue()
	{
		return getSingleInput().getValue();
	}

	/**
	 * @see org.ggp.base.util.propnet.architecture.Component#toString()
	 */
	@Override
	public String toString()
	{
		return "TRANSITION";
	}

	/**
	 * @see org.ggp.base.util.propnet.architecture.Component#toDot()
	 */
	@Override
	public String toDot()
	{
		return toDot("box", getValue() ? "red" : "grey", "TRANSITION");
	}

	@Override
	public String bitString(int cValue) {
		// TODO Auto-generated method stub
		return toDot("box", cValue != 0 ? "red" : "grey", "TRANSITION");
	}
}
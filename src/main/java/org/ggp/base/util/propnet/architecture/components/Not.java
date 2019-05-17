package org.ggp.base.util.propnet.architecture.components;

import org.ggp.base.util.propnet.architecture.Component;

/**
 * The Not class is designed to represent logical NOT gates.
 */
@SuppressWarnings("serial")
public final class Not extends Component
{
	/**
	 * Returns the inverse of the input to the not.
	 *
	 * @see org.ggp.base.util.propnet.architecture.Component#getValue()
	 */
	@Override
	public boolean getValue()
	{
		return !getSingleInput().getValue();
	}

	/**
	 * @see org.ggp.base.util.propnet.architecture.Component#toString()
	 */
	@Override
	public String toString()
	{
		return "NOT";
	}

	/**
	 * @see org.ggp.base.util.propnet.architecture.Component#toDot()
	 */
	@Override
	public String toDot()
	{
		return toDot("invtriangle", "grey", "NOT");
	}

	@Override
	public String bitString(int cValue) {
		// TODO Auto-generated method stub
		return null;
	}
}
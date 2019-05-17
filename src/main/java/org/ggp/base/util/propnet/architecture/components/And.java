package org.ggp.base.util.propnet.architecture.components;

import org.ggp.base.util.propnet.architecture.Component;

/**
 * The And class is designed to represent logical AND gates.
 */
@SuppressWarnings("serial")
public final class And extends Component
{
	/**
	 * Returns true if and only if every input to the and is true.
	 *
	 * @see org.ggp.base.util.propnet.architecture.Component#getValue()
	 */
	@Override
	public boolean getValue()
	{
		for ( Component component : getInputs() )
		{
			if ( !component.getValue() )
			{
				return false;
			}
		}
		return true;
	}

	/**
	 * @see org.ggp.base.util.propnet.architecture.Component#toString()
	 */
	@Override
	public String toString()
	{
		return "AND";

	}

	/**
	 * @see org.ggp.base.util.propnet.architecture.Component#toDot()
	 */
	@Override
	public String toDot()
	{
		return toDot("invhouse", "grey", "AND");
	}

	@Override
	public String bitString(int cValue) {
		// TODO Auto-generated method stub
		return null;
	}

}

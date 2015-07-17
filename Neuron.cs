/*

MIT license:

Copyright (c) 2015 AngryAnt, Emil Johansen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/


using System;


namespace Learn
{
	public class Neuron
	{
		double[] m_Inputs, m_Weights;
		double m_BiasWeight, m_Output;
		bool m_CalculateOutput = true;
		Random m_Random = new Random ();


		public double BiasWeight
		{
			get
			{
				return m_BiasWeight;
			}
			set
			{
				m_BiasWeight = value;

				m_CalculateOutput = true;
			}
		}


		public int InputCount
		{
			get
			{
				return m_Inputs.Length;
			}
		}


		public double Output
		{
			get
			{
				if (m_CalculateOutput)
				{
					double weightedInput = 0;
					for (int input = 0; input < m_Inputs.Length; ++input)
					{
						weightedInput += m_Inputs[input] * m_Weights[input];
					}

					m_Output = Sigmoid.Output (weightedInput + BiasWeight);

					m_CalculateOutput = false;
				}

				return m_Output;
			}
		}


		public Neuron (int inputCount)
		{
			m_Inputs = new double[inputCount];
			m_Weights = new double[inputCount];
			BiasWeight = 0;
		}


		public double GetWeight (int index)
		{
			return m_Weights[index];
		}


		public void SetWeight (int input, double value)
		{
			m_Weights[input] = value;
			m_CalculateOutput = true;
		}


		public void SetInput (int input, double value)
		{
			m_Inputs[input] = value;
			m_CalculateOutput = true;
		}


		public void RandomizeWeights ()
		{
			for (int input = 0; input < m_Inputs.Length; ++input)
			{
				m_Weights[input] = m_Random.NextDouble ();
			}

			BiasWeight = m_Random.NextDouble ();

			m_CalculateOutput = true;
		}
	}
}

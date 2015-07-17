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


namespace Learn
{
	public partial class BackPropagation
	{
		class BPNeuron
		{
			// The wrapped neuron
			Neuron m_Neuron;
			// The calculated error of this neuron and the update to add to its bias value (stored for use with momentum)
			double m_Error, m_BiasUpdate;
			// The calculated weight updates to add to the weights of the neuron (stored for use with momentum)
			double[] m_WeightUpdates;


			public double Output
			{
				get
				{
					return m_Neuron.Output;
				}
			}


			public double Error
			{
				get
				{
					return m_Error;
				}
			}


			public BPNeuron (Neuron neuron)
			{
				m_Neuron = neuron;
				m_WeightUpdates = new double[neuron.InputCount];
			}


			public double GetWeight (int index)
			{
				return m_Neuron.GetWeight (index);
			}


			// Given derived error value, calculate final neuron error
			public void CalculateError (double derivedError)
			{
				m_Error = derivedError * Sigmoid.Derivative (Output);
			}


			// Update neuron bias and weights to compensate for calculated error on given input
			public void Update (double[] input, double learningRate, double momentum)
			{
				m_BiasUpdate =
					m_Error * learningRate +						// Base bias update
					momentum * m_BiasUpdate;						// Momentum

				m_Neuron.BiasWeight += m_BiasUpdate;				// Assign

				for (int index = 0; index < m_WeightUpdates.Length; ++index)
				{
					m_WeightUpdates[index] =
						m_Error * input[index] * learningRate +		// Base weight update
						momentum * m_WeightUpdates[index];			// Momentum

					// Assign
					m_Neuron.SetWeight (index, m_Neuron.GetWeight (index) + m_WeightUpdates[index]);
				}
			}
		}
	}
}

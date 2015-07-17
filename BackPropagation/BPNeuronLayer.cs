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
using System.Linq;


namespace Learn
{
	public partial class BackPropagation
	{
		class BPNeuronLayer
		{
			// The wrapped neurons of this wrapped layer
			BPNeuron[] m_Neurons;
			// The neighbouring layers
			BPNeuronLayer m_Previous, m_Next;


			public BPNeuronLayer (NeuronLayer layer, BPNeuronLayer previous)
			{
				m_Neurons = layer.Select (n => new BPNeuron (n)).ToArray ();
				m_Previous = previous;
			}


			public BPNeuronLayer SetNext (BPNeuronLayer next)
			{
				m_Next = next;

				return next;
			}


			// Calculate error values for all layer neurons, then ask the previous layer (if available) to do the same
			public void CalculateError (double[] derivedErrors)
			{
				// Apply the derived errors
				for (int index = 0; index < m_Neurons.Length; ++index)
				{
					m_Neurons[index].CalculateError (derivedErrors[index]);
				}

				// If this is the first layer, there's nothing more to be done
				if (m_Previous == null)
				{
					return;
				}

				// Calculate the weighted error derived from this layer to the previous
				m_Previous.CalculateWeightedErrorFromNext (derivedErrors);

				// Let the previous layer handle that derived error
				m_Previous.CalculateError (derivedErrors);
			}


			// Update all layer neurons to compensate for calculated error, then ask the next layer (if available) to do the same
			public void Update (double[] input, double learningRate, double momentum)
			{
				// Calculate updates for all neurons in this layer
				for (int index = 0; index < m_Neurons.Length; ++index)
				{
					m_Neurons[index].Update (input, learningRate, momentum);
				}

				// If this is the last layer, there's nothing more to be done
				if (m_Next == null)
				{
					return;
				}

				// Calculate input of the next layer as the output of this one
				CollectOutput (input);

				// Calculate update for the next layer
				m_Next.Update (input, learningRate, momentum);
			}


			// Calculate the difference between the layer output and desired result, store it in outputDiff, and return layer error
			public double CalculateOutputDiff (double[] desiredResult, double[] outputDiff)
			{
				double diff = 0, error = 0;

				// Get output diff for each neuron
				for (int index = 0; index < m_Neurons.Length; ++index)
				{
					// Calculate the difference between desired neuron output and actual output
					outputDiff[index] = diff = desiredResult[index] - m_Neurons[index].Output;

					// Accumulate layer error
					error += diff * diff;
				}

				// Return layer error
				return error / 2.0;
			}


			// Collect the layer output into the given array
			void CollectOutput (double[] output)
			{
				for (int index = 0; index < m_Neurons.Length; ++index)
				{
					output[index] = m_Neurons[index].Output;
				}
			}


			// Calculate the error value from the next layer, scaled by weights, and store it in weightedError
			void CalculateWeightedErrorFromNext (double[] weightedError)
			{
				// Get weighted error from next layer for each neuron
				for (int neuronIndex = 0; neuronIndex < m_Neurons.Length; ++neuronIndex)
				{
					double error = 0;

					// Accumulate weighted error from each neuron in the next layer
					for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < m_Next.m_Neurons.Length; ++nextLayerNeuronIndex)
					{
						double
							nextLayerNeuronError = m_Next.m_Neurons[nextLayerNeuronIndex].Error,
							neuronToNeuronWeight = m_Next.m_Neurons[nextLayerNeuronIndex].GetWeight (neuronIndex);

						// The weighted error from the neuron in the next layer is its error scaled by the weight between it and the current layer neuron
						error += nextLayerNeuronError * neuronToNeuronWeight;
					}

					// Store accumulated weighted error for this neuron
					weightedError[neuronIndex] = error;
				}
			}
		}
	}
}

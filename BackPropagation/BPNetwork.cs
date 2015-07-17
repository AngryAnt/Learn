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
	public partial class BackPropagation
	{
		class BPNetwork
		{
			// The wrapped network
			Network m_Network;
			// The first and last wrapped layers
			BPNeuronLayer m_First, m_Last;
			// This buffer is passed between layer functions and resized as needed
			double[] m_TrainingBuffer;


			public BPNetwork (Network network)
			{
				m_Network = network;
				m_Last = m_First = new BPNeuronLayer (m_Network[0], null);

				for (int index = 1; index < m_Network.LayerCount; ++index)
				{
					m_Last = m_Last.SetNext (new BPNeuronLayer (m_Network[index], m_Last));
				}

				m_TrainingBuffer = new double[Math.Max (m_Network.MaxWidth, m_Network.InputCount)];
			}


			public double Train (double[] input, double[] output, BackPropagation trainer)
			{
				// Run the given input throug the network in order to obtain output
				m_Network.Run (input);
				// Calculate network error between the generated output and the expected output
				double error = CalculateError (output);
				// Update the network to remedy the calculated error
				Update (input, trainer.LearningRate, trainer.Momentum);

				return error;
			}


			double CalculateError (double[] desiredResult)
			{
				// Store the diff between the generated output and the desired one in the training buffer
				double error = m_Last.CalculateOutputDiff (desiredResult, m_TrainingBuffer);

				// Pass the training buffer content with diff as the derived errors for CalculateError on the last layer (this progresses back through the other layers)
				m_Last.CalculateError (m_TrainingBuffer);

				return error;
			}


			void Update (double[] input, double learningRate, double momentum)
			{
				// Store input in the training buffer
				Array.Copy (input, m_TrainingBuffer, m_Network.InputCount);

				// Pass the training bugger as the input for Update on the first layer (this progresses forward through the other layers)
				m_First.Update (m_TrainingBuffer, learningRate, momentum);
			}
		}
	}
}

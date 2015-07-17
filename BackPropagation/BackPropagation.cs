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
		BPNetwork m_Trainer;
		Network m_Network;
		double m_LearningRate = .1, m_Momentum = .1;


		public double LearningRate
		{
			get
			{
				return m_LearningRate;
			}
			set
			{
				m_LearningRate = Math.Max (0, Math.Min (value, 1));
			}
		}


		public double Momentum
		{
			get
			{
				return m_Momentum;
			}
			set
			{
				m_Momentum = Math.Max (0, Math.Min (value, 1));
			}
		}


		// A training instance is capable of training a single given network
		public BackPropagation (Network network)
		{
			m_Network = network;
			m_Trainer = new BPNetwork (network);
		}


		// Train with multiple input/output sets, returning the total error
		public double TrainSet (double[][] input, double[][] output)
		{
			// Verify matching training set length
			if (input.Length != output.Length)
			{
				throw new ArgumentException (string.Format (
					"Input/output training set length mismatch: Input = {0}, output = {1}",
					input.Length,
					output.Length
				));
			}

			double error = 0;

			for (int index = 0; index < input.Length; ++index)
			{
				error += Train (input[index], output[index]);
			}

			return error;
		}


		// Perform a single training run, receiving given input and expecting given output - resulting error value is returned
		public double Train (double[] input, double[] output)
		{
			// Verify training set input length
			if (input.Length < m_Network.InputCount)
			{
				throw new ArgumentException (string.Format (
					"Provided training input is too short: Training = {0}, network = {1}",
					input.Length,
					m_Network.InputCount
				));
			}
			// Verify training set output length
			if (output.Length < m_Network.OutputCount)
			{
				throw new ArgumentException (string.Format (
					"Provided training output is too short: Training = {0}, network = {1}",
					output.Length,
					m_Network.OutputCount
				));
			}

			// Train
			return m_Trainer.Train (input, output, this);
		}
	}
}

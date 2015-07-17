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
using System.Collections;
using System.Collections.Generic;


namespace Learn
{
	public class Network : IEnumerable<NeuronLayer>
	{
		NeuronLayer[] m_Layers;
		int m_MaxWidth;
		double[] m_OutputBuffer;


		public int InputCount
		{
			get
			{
				return m_Layers[0].InputCount;
			}
		}


		public int OutputCount
		{
			get
			{
				return m_Layers[m_Layers.Length - 1].NeuronCount;
			}
		}


		public int MaxWidth
		{
			get
			{
				return m_MaxWidth;
			}
		}


		public int LayerCount
		{
			get
			{
				return m_Layers.Length;
			}
		}


		public NeuronLayer this[int index]
		{
			get
			{
				return m_Layers[index];
			}
		}


		public Network (int inputCount, params int[] layerSizes)
		{
			// Validate input count
			if (inputCount < 1)
			{
				throw new ArgumentException ("Invalid input count: " + inputCount);
			}

			// If nothing is specified, default layer setup to one layer of inputCount neurons
			if (layerSizes.Length < 1)
			{
				layerSizes = new int[] {inputCount};
			}

			// Create the layers
			m_Layers = new NeuronLayer[layerSizes.Length];
			m_Layers[0] = NeuronLayer.CreateInputLayer (layerSizes[0], inputCount);
			m_MaxWidth = layerSizes[0];
			for (int layerIndex = 1; layerIndex < layerSizes.Length; ++layerIndex)
			{
				int size = layerSizes[layerIndex];

				if (size < 1)
				{
					throw new ArgumentException (string.Format ("Invalid size of layer {0}: {1}", layerIndex, size));
				}

				m_MaxWidth = Math.Max (m_MaxWidth, size);

				m_Layers[layerIndex] = NeuronLayer.CreateLayer (size, m_Layers[layerIndex - 1]);
			}

			// Having identified the max width of the network, we can now create the output buffer
			m_OutputBuffer = new double[m_MaxWidth];
		}


		// Assign [0-1] random values to all network weigths
		public void RandomizeWeights ()
		{
			for (int index = 0; index < m_Layers.Length; ++index)
			{
				m_Layers[index].RandomizeWeights ();
			}
		}


		// Run the given input through the network and optionally have the network output deposited into the given array
		// NOTE: Input and output arrays may be longer than network InputCount and OutputCount respectively
		public void Run (double[] input, double[] output = null)
		{
			int inputCount = InputCount, outputCount = OutputCount;

			// Validate input array length
			if (input.Length < inputCount)
			{
				throw new ArgumentException (string.Format ("Too short input array: Network = {0}, provided = {1}", inputCount, input.Length));
			}
			// Validate output array length
			if (output != null && output.Length < outputCount)
			{
				throw new ArgumentException (string.Format ("Too short output array: Network = {0}, provided = {1}", outputCount, output.Length));
			}

			// Run each layer, passing network input to the first, then passing output of the layer as input of the next
			for (int index = 0; index < m_Layers.Length; ++index)
			{
				m_Layers[index].Run (input, m_OutputBuffer);
				input = m_OutputBuffer;
			}

			// Copy output if receiving
			if (output != null)
			{
				Array.Copy (m_OutputBuffer, output, outputCount);
			}
		}


		IEnumerator IEnumerable.GetEnumerator ()
		{
			return m_Layers.GetEnumerator ();
		}


		IEnumerator<NeuronLayer> IEnumerable<NeuronLayer>.GetEnumerator ()
		{
			foreach (NeuronLayer layer in m_Layers)
			{
				yield return layer;
			}
		}
	}
}

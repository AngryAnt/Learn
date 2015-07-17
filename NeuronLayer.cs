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
	public class NeuronLayer : IEnumerable<Neuron>
	{
		Neuron[] m_Neurons;


		public int InputCount
		{
			get
			{
				return m_Neurons[0].InputCount;
			}
		}


		public int NeuronCount
		{
			get
			{
				return m_Neurons.Length;
			}
		}


		public Neuron this[int index]
		{
			get
			{
				return m_Neurons[index];
			}
		}


		protected NeuronLayer (int neuronCount, int inputCount)
		{
			m_Neurons = new Neuron[neuronCount];
			for (int index = 0; index < neuronCount; ++index)
			{
				m_Neurons[index] = new Neuron (inputCount);
			}
		}


		public static NeuronLayer CreateInputLayer (int neuronCount, int inputCount)
		{
			return new NeuronLayer (neuronCount, inputCount);
		}


		public static NeuronLayer CreateLayer (int neuronCount, NeuronLayer preceedingLayer)
		{
			return new NeuronLayer (neuronCount, preceedingLayer.NeuronCount);
		}


		public void RandomizeWeights ()
		{
			for (int index = 0; index < m_Neurons.Length; ++index)
			{
				m_Neurons[index].RandomizeWeights ();
			}
		}


		public void Run (double[] input, double[] output)
		{
			for (int neuronIndex = 0; neuronIndex < m_Neurons.Length; ++neuronIndex)
			{
				Neuron neuron = m_Neurons[neuronIndex];
				for (int inputIndex = 0; inputIndex < neuron.InputCount; ++inputIndex)
				{
					neuron.SetInput (inputIndex, input[inputIndex]);
				}
			}

			for (int index = 0; index < m_Neurons.Length; ++index)
			{
				output[index] = m_Neurons[index].Output;
			}
		}


		IEnumerator IEnumerable.GetEnumerator ()
		{
			return m_Neurons.GetEnumerator ();
		}


		IEnumerator<Neuron> IEnumerable<Neuron>.GetEnumerator ()
		{
			foreach (Neuron neuron in m_Neurons)
			{
				yield return neuron;
			}
		}
	}
}

Readme
======
Because I couldn't find a decent C# neural net back-propagation implementation with non-contageous licensing.

Pull requests are welcome.

Creating a network
------------------
    using Learn;

    // ...

	// Two inputs, one hidden layer with two neurons, one neuron in the second (and output) layer
    Network network = new Network (2, 2, 1);
    network.RandomizeWeights ();

Running a network
-----------------

    network.Run (inputBuffer, outputBuffer);

Training a network
------------------
    using Learn;

    // ...

    BackPropagation teacher = new BackPropagation (network)
    {
    	LearningRate = .3,
    	Momentum = .9
    };

    double[][]
    	input = new double[][]
    	{
    		new[] {0.0, 0.0},
    		new[] {0.0, 1.0},
    		new[] {1.0, 0.0},
    		new[] {1.0, 1.0}
    	},
    	output = new double[][]
    	{
    		new[] {0.0},
    		new[] {1.0},
    		new[] {1.0},
    		new[] {0.0}
    	};

    double targetError = .01;
    double error = targetError;
    int iteration, maxIterations = 5000;

    for (iteration = 0; iteration < maxIterations && targetError < error; ++iteration)
    {
    	error = teacher.TrainSet (input, output);
    }

Todo
====
 * Test what happens if training input or output is negative. Something we should guard against or are the results useful?
 * Test with too large input and output buffers - verify that it works.
 * Do a simplification pass.
 * Intellisense comments?
 * Serialisation interface.

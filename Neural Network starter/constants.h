#pragma once
struct Constants {
	const static int NUMBER_OF_HIDDEN_LAYERS = 1;
	const static int NEURONS_PER_HIDDEN_LAYER = 30;

	// Interation after to compute Stochastic Gradient Descent
	const static int MINI_BATCH_SIZE = 10;

	constexpr static long double LEARNING_RATE = 3.0;
	constexpr static bool SHOULD_WRITE_TO_FILE = true;
};
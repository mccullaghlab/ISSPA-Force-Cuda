sudo nvprof -f -o algorithm_3.timeline ../bin/total_force_cuda.x config.txt
sudo nvprof -f --analysis-metrics -o algorithm_3.metrics ../bin/total_force_cuda.x config.txt

sudo nvprof -o isspa2_analysis.timeline ../../bin/total_force_cuda.x config.txt
sudo nvprof --analysis-metrics -o isspa2_analysis.metrics ../../bin/total_force_cuda.x config.txt
#sudo nvprof -f --analysis-metrics -o isspa2_analysis.nvprof ../../bin/total_force_cuda.x config.txt

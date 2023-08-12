
import engine

# engine.do_nas_search("D:/experiments", config_file='config/config.json', grammar_file='config/lenet.grammar')

for run in range(0, 10):
	print(f"***** Starting FDENSER{run:02d} of 10, random seed {100+run} **********")
	engine.do_nas_search("D:/experiments/FDENSER10", config_file='config/config_fdenser.json', grammar_file='config/lenet.grammar', override_experiment_name=f"FDENSER{run:02d}", override_random_seed=100+run)

#for run in range(0, 10):
#	print(f"***** Starting Stepper{run:02d} of 10, random seed {100+run} **********")
#	engine.do_nas_search("D:/experiments/Stepper10", config_file='config/config10.json', grammar_file='config/lenet.grammar', override_experiment_name=f"Stepper{run:02d}", override_random_seed=100+run)

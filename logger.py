class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def init_logger(file_path=None, overwrite=True):
	global logging_file_path
	global logging_line_list
	global logging_overwrite
	global logging_append_to_line

	logging_file_path = file_path
	logging_line_list = []
	logging_overwrite = overwrite
	logging_append_to_line = False
	logger_configuration()

def logger_configuration(log_training=True, log_mutations=True):
	global logging_training
	global logging_mutations
	logging_training = log_training
	logging_mutations = log_mutations

def log_append(value):
	global logging_append_to_line
	if logging_append_to_line:
		logging_line_list[-1] += str(value)
		logging_append_to_line = False
	else:
		logging_line_list.append(str(value))

def log_append_flush(value):
	log_append(value)
	log_flush()

def log_append_nolf(value):
	global logging_append_to_line
	log_append(value)
	logging_append_to_line = True

def log(value=''):
	print(value)
	log_append_flush(value)

def log_noflush(value):
	print(value)
	log_append(value)

def log_nolf(value):
	print(value, end='')
	log_append_nolf(value)

def log_mutation(value):
	if logging_mutations:
		print(bcolors.OKGREEN + value  + bcolors.ENDC)
		log_append(value)


def log_training(value):
	if logging_training:
		print(bcolors.OKBLUE + value  + bcolors.ENDC)
		log_append_flush(value)

def log_training_nolf(value):
	if logging_training:
		print(bcolors.OKBLUE + value  + bcolors.ENDC, end='')
		log_append_nolf(value)

def log_bold(value):
	print(bcolors.BOLD + value + bcolors.ENDC)
	log_append_flush(value)

def log_warning(value):
	print(bcolors.FAIL + '*** ' + value + ' ***' + bcolors.ENDC)
	log_append_flush(value)

def log_flush():
	global logging_overwrite
	if len(logging_line_list):
		if logging_file_path:
			with open(logging_file_path, 'w' if logging_overwrite else 'a') as file:
				for line in logging_line_list:
					file.write(line + '\n')
		logging_line_list.clear()
		logging_overwrite = False

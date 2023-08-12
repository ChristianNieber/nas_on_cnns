logging_file_path = ''
logging_line_list = []
logging_overwrite = False
logging_append_to_line = False
logging_training = False
logging_mutations = False
logging_debug = False


class TerminalColors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


def init_logger(file_path=None):
	global logging_file_path
	global logging_line_list
	global logging_overwrite
	global logging_append_to_line

	logging_file_path = file_path
	logging_line_list = []
	logging_overwrite = True
	logging_append_to_line = False
	logger_configuration()

def logger_configure_overwrite(overwrite):
	global logging_overwrite
	logging_overwrite = overwrite

def logger_configuration(logger_log_training=True, logger_log_mutations=True, logger_log_debug=False):
	global logging_training
	global logging_mutations
	global logging_debug
	logging_training = logger_log_training
	logging_mutations = logger_log_mutations
	logging_debug = logger_log_debug

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
		print(TerminalColors.OKGREEN + value + TerminalColors.ENDC)
		log_append(value)


def log_training(value):
	if logging_training:
		print(TerminalColors.OKBLUE + value + TerminalColors.ENDC)
		log_append_flush(value)


def log_training_nolf(value):
	if logging_training:
		print(TerminalColors.OKBLUE + value + TerminalColors.ENDC, end='')
		log_append_nolf(value)


def log_bold(value):
	print(TerminalColors.BOLD + value + TerminalColors.ENDC)
	log_append_flush(value)


def log_warning(value):
	print(TerminalColors.FAIL + '*** ' + value + ' ***' + TerminalColors.ENDC)
	log_append_flush(value)

def log_debug(value):
	if logging_debug:
		print(TerminalColors.OKCYAN + value + TerminalColors.ENDC)
		log_append(value)

def log_flush():
	global logging_overwrite
	if len(logging_line_list):
		if logging_file_path:
			with open(logging_file_path, 'w' if logging_overwrite else 'a', encoding="utf-8") as file:
				for line in logging_line_list:
					file.write(line + '\n')
		logging_line_list.clear()
		logging_overwrite = False

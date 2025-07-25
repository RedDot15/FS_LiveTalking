import logging
 
# Configure Logger
# Gets or creates a logger named after the current module.
logger = logging.getLogger(__name__)
# Sets the logger's minimum logging level to DEBUG.
logger.setLevel(logging.DEBUG)
# Creates a formatter to define the log message format.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Creates a file handler to write logs to a file named 'livetalking.log'.
# It can be changed to use a StreamHandler to output to the console, or a combination of multiple Handlers.
fhandler = logging.FileHandler('livetalking.log')  # It can be changed to use a StreamHandler to output to the console, or a combination of multiple Handlers.
# Sets the formatter for the file handler.
fhandler.setFormatter(formatter)
# Sets the file handler's minimum logging level to INFO.
fhandler.setLevel(logging.INFO)
# Adds the file handler to the logger.
logger.addHandler(fhandler)

# handler = logging.StreamHandler() # Creates a stream handler to output to the console.
# handler.setLevel(logging.DEBUG) # Sets the stream handler's minimum logging level to DEBUG.
# sformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Creates a specific formatter for the stream handler.
# handler.setFormatter(sformatter) # Sets the formatter for the stream handler.
# logger.addHandler(handler) # Adds the stream handler to the logger.
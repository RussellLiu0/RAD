import logging

def init_logging(log_path):

    logging.basicConfig(
                    level    = logging.DEBUG,
                    format   = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt  = '%m-%d %H:%M',
                    filename = log_path, # given filename will omit console output handler
                    filemode = 'a')
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s LINE %(lineno)-4d : %(levelname)-8s %(message)s')
    # 控制 console的输出，这个输出更为简单。
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return 

def init_log(path: str='log.log'):
    init_logging(path)
    return 

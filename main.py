import argparse

# Argument parser
parser = argparse.ArgumentParser(
                    prog='Peppermint Conversational Framework launching script.',
                    description='Will launch workers according to the config file passed as '
                                'the first positional argument. Press ^C to exit. '
                                'Logs can be found in the logs/ directory')
parser.add_argument('config_file')
parser.add_argument('-t', '--timeout', type=int, default=120,
                    help='Timeout for workers to be ready, in seconds, default: 120')
parser.add_argument("-d", "--debug", action='store_true', help="print debug logs in addition log file")


if __name__ == "__main__":
    args = parser.parse_args()
    READY_TIMEOUT = args.timeout

    # Imports
    import os
    import time
    import yaml
    import logging
    import multiprocessing as mp

    from ppm.worker import WorkerProcess
    from ppm.logs import logger_process, mp_logger

    # Import Workers' files to run registration
    from ppm.io import stream, transcript_to_file, recording
    from ppm.ai import vad, asr, faq, tts
    from ppm.network import sockets

    # Logger
    log_q = mp.Manager().Queue()
    log_process = mp.Process(target=logger_process, name='log_process', args=(log_q, args.debug))
    log_process.start()
    print('Main and log processes started', mp.current_process().pid, log_process.pid)
    logger = mp_logger(log_q, name='ppm')
    logger.info("-" * 10 + "New Session" + "-" * 10)

    processes = None

    try:

        # Load config
        if not os.path.isfile(args.config_file):
            raise FileNotFoundError(f'{args.config_file} not found')
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f'Using {args.config_file}')

        # Taking care of global arguments
        global_params = config.pop('global', {})

        # Initialize worker processes
        processes = {
            worker_type: WorkerProcess.from_config(worker_type, worker_config, global_params)
            for worker_type, worker_config in config.items()
        }

        # Setting up queues and events
        exit_event, start_event = mp.Manager().Event(), mp.Manager().Event()    # will be used to send exit and start signal to processes
        ready_events, done_events = [], []                  # used by workers to signal that they're ready or done
        for worker_key, worker in processes.items():
            # Queues
            if config[worker_key] is not None:
                to = config[worker_key].get("to")
                if to is not None:
                    to = [to] if type(to) is not list else to
                    # Creating and assigning queues
                    for target_worker in to:
                        q = mp.Manager().Queue()
                        # Source worker
                        worker.output_queues.append(q)
                        # Target worker
                        if processes[target_worker].input_queue is not None:
                            raise Exception(f'Only one input queue per worker is allowed ({target_worker})')
                        processes[target_worker].input_queue = q
            worker.log_queue = log_q

            # Events
            worker.exit_event, worker.start_event = exit_event, start_event
            worker.ready, worker.done = mp.Manager().Event(), mp.Manager().Event()
            ready_events.append(worker.ready)
            done_events.append(worker.done)

        # Start processes
        logger.info("Starting processes.. ")
        for p in processes.values():
            p.start()
            logger.info(f'{p.name}:{p.pid}')
        logger.info("All processes started.")

        # Wait for workers to be ready
        logger.info("Waiting for workers to be ready.. ")
        ready_check_count = 0
        while any(not e.is_set() for e in ready_events) and ready_check_count <= READY_TIMEOUT:
            ready_check_count += 1
            time.sleep(1)
        if ready_check_count >= READY_TIMEOUT:
            raise TimeoutError(f'{[w.name for w in processes.values() if not w.ready.is_set()]} not ready in time.')
        start_event.set()
        logger.info("All workers ready.")

        done_workers = set()

        # Run until KeyboardInterrupt
        while True:
            try:
                # Checking on child processes
                # Alive
                if any(not p.is_alive() for p in processes.values()):
                    deads = [p.name for p in processes.values() if not p.is_alive()]
                    logger.error(f"Some process(es) died {deads}, quitting.")
                    break
                # Done
                for key, worker in processes.items():
                    if worker.done.is_set() and key not in done_workers:
                        done_workers.add(key)
                        logger.info(f"Worker {worker.name} is done.")
                time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("Ctrl-C detected, quitting...")
                break

        # Sending exit signal to all processes in case they didn't receive KeyboardInterrupt
        exit_event.set()

    except Exception as e:
        logger.exception(e)

    finally:
        # Terminating processes
        if processes is not None:
            logger.info("Joining processes...")
            for p in processes.values():
                if p.is_alive():
                    p.join(timeout=3)
                    if p.is_alive():
                        p.terminate()
                        logger.error(f"! Had to terminate {p.name}, {p.pid}")
            logger.info("Done.")

        # Log Process
        log_q.put(None)     # Signal log_process to quit
        log_process.join(timeout=3)
        if log_process.is_alive():
            log_process.terminate()

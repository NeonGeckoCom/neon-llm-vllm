from rmq import ChatgptMQ



def main():
    # Run RabbitMQ
    chatgptMQ = ChatgptMQ()
    chatgptMQ.run(run_sync=False, run_consumers=True,
                      daemonize_consumers=True)
    chatgptMQ.observer_thread.join()



if __name__ == "__main__":
    main()
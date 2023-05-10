from rmq import ChatgptMQ



def main():
    # Run RabbitMQ
    chatgptMQ = ChatgptMQ()
    chatgptMQ.run(run_sync=False, run_consumers=True,
                      daemonize_consumers=True)



if __name__ == "__main__":
    main()
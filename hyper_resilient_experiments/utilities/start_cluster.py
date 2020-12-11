"""Utility script used to start up a ray tune cluster on """
import os
import argparse

def start_cluster(yaml="/home/mzvyagin/default_cluster.yaml", cluster_name="default"):
    # get head node
    host = os.popen('hostname').read().strip()
    # get sub nodes
    # know how: https://www.alcf.anl.gov/support-center/theta/faqs-queueing-and-running-theta#one
    nodefile = os.environ['COBALT_NODEFILE']
    f = open(nodefile, "r")
    nodes = f.readlines()
    num_workers = len(nodes)
    for x in nodes:
        if host in x:
            nodes.remove(x)
    workers = ""
    authorization = '{ssh_user: mzvyagin}'
    print("INCORRECT IMPLEMENTATION FOR NUM WORKERS")
    lines = []
    lines.append(
        f'''
        cluster_name: default
        provider:
            type: local
            head_ip: {host}
            worker_ips: {workers}
        auth: {authorization}
        min_workers = {num_workers}
        max_workers = {num_workers}
        setup_commands:
            - source ~/.bashrc; conda activate dl
        '''
    )
    f = open(yaml, "w+")
    f.write(lines)
    os.popen('ray up '+yaml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', help="Absolute path for yaml file.")
    parser.add_argument('-c', '--cluster', help="Specify name for cluster.")
    args = parser.parse_args()
    if args.yaml:
        yaml = args.yaml
    else:
        yaml = "/home/mzvyagin/default_cluster.yaml"
    if args.cluster:
        cluster = args.cluster
    else:
        cluster = "default"
    start_cluster(yaml, cluster)

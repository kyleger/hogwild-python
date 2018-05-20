#!/usr/bin/env bash

APP_NAME=hogwild
REPO=liabifano
KUBER_LOGIN=cs449g9

DATA_PATH=data/datasets

while getopts ":n:r:" opt; do
  case $opt in
    n) N_WORKERS="$OPTARG";;
    r) RUNNING_MODE="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


function shutdown_infra {
    echo
    echo "----- Deleting remaining services and configurations -----"
    if ! [[ -z $(kubectl get services | grep workers-service) ]];
    then
        kubectl delete -f Kubernetes/workers.yaml --cascade=true
    fi;

    if ! [[ -z $(kubectl get services | grep coordinator-service) ]];
    then
        kubectl delete -f Kubernetes/coordinator.yaml --cascade=true
    fi;

    if ! [[ -z $(kubectl get configmap | grep hogwild-config) ]];
    then
        kubectl delete configmap hogwild-config
    fi;
}


# Don't forget to run login first with `docker login`
docker login --username=$DOCKER_USER --password=$DOCKER_PASS

shutdown_infra

echo
echo "----- Building and Pushing docker to Docker Hub -----"
docker build -f `pwd`/Docker/Dockerfile `pwd` -t ${REPO}/${APP_NAME}
docker push ${REPO}/${APP_NAME}

echo
echo "----- Starting workers -----"
kubectl create configmap hogwild-config --from-literal=replicas=${N_WORKERS} \
                                        --from-literal=running_mode=${RUNNING_MODE} \
                                        --from-literal=data_path=${DATA_PATH}
sed "s/\(replicas:\)\(.*\)/\1 ${N_WORKERS}/" Kubernetes/workers_template.yaml > Kubernetes/workers.yaml
kubectl create -f Kubernetes/workers.yaml

while [ $(kubectl get pods | grep worker | grep Running | wc -l) != ${N_WORKERS} ]
do
    sleep 1
done

echo
echo "----- Workers are up and running, starting coordinator -----"
kubectl create -f Kubernetes/coordinator.yaml

#shutdown_infra

# useful commands
#kubectl delete po,svc --all
#docker build -f `pwd`/Docker/coordinator/Dockerfile `pwd` -t ${REPO}/${APP_NAME}_coordinator
#docker push ${REPO}/hogwild_coordinator
## Create `pod` in the cluster
#kubectl create -f ./Kubernetes/hog-pod.yaml
#kubectl describe pod/${APP_NAME} -n ${KUBER_LOGIN}
# kubectl logs $APP_NAME -p --container="coordinator"
#kubectl -n my-ns delete po,svc --all
#kubectl delete -f Kubernetes/workers_template.yaml --cascade=true
#kubectl exec -it coordinator-0 -- /bin/bash
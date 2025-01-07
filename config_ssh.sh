#!/bin/bash

STAGE_DIR=/tmp
export SSH_PORT=8081

cat /etc/ssh/sshd_config > ${STAGE_DIR}/sshd_config && \
	sed "0,/^Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config
cat /etc/ssh/ssh_config > ${STAGE_DIR}/ssh_config && \
	sed "0,/^#   Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/ssh_config > /etc/ssh/ssh_config
echo "PermitRootLogin yes\n" >> /etc/ssh/sshd_config && \
	echo " StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
	echo " UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config
mkdir -p /root/.ssh && ssh-keygen -t rsa -f ~/.ssh/id_rsa -P '' && cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
echo "Done."


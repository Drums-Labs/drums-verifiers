#!/usr/bin/env bash
# launch-host.sh: a fresh acceptance host (latest Ubuntu 24.04 from SSM, c8i.4xlarge, 120 GiB gp3), plus an ssh alias
# `drums-acceptance`. Prints the instance id. Terminate it after the run:
#   aws --profile $PROFILE --region us-east-1 ec2 terminate-instances --instance-ids <id>
#   PROFILE=drums-operator SUBNET=subnet-0bc305a757e1d96ff SG=sg-0bec95602db3fd3b5 KEY=drums-ec2-mac acceptance/launch-host.sh
set -euo pipefail
P=${PROFILE:-drums-operator}; REG=${REGION:-us-east-1}; TYPE=${TYPE:-c8i.4xlarge}
SUBNET=${SUBNET:?}; SG=${SG:?}; KEY=${KEY:?}; PEM=${PEM:-~/.ssh/$KEY.pem}
AMI=$(aws --profile "$P" --region "$REG" ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
ID=$(aws --profile "$P" --region "$REG" ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" --subnet-id "$SUBNET" \
  --security-group-ids "$SG" --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3,Iops=6000,Throughput=500,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=drums-acceptance}]' --query 'Instances[].InstanceId' --output text)
aws --profile "$P" --region "$REG" ec2 wait instance-running --instance-ids "$ID"
IP=$(aws --profile "$P" --region "$REG" ec2 describe-instances --instance-ids "$ID" --query 'Reservations[].Instances[].PublicIpAddress' --output text)
grep -q '^Host drums-acceptance$' ~/.ssh/config || printf '\nHost drums-acceptance\n  HostName %s\n  User ubuntu\n  IdentityFile %s\n  IdentitiesOnly yes\n  ServerAliveInterval 30\n  StrictHostKeyChecking accept-new\n' "$IP" "$PEM" >> ~/.ssh/config
python3 - "$IP" <<'EOF'
import os, re, sys
p = os.path.expanduser("~/.ssh/config"); s = open(p).read()
s = re.sub(r"(Host drums-acceptance\n  HostName )\S+", r"\g<1>" + sys.argv[1], s); open(p, "w").write(s)
EOF
for i in $(seq 1 30); do ssh -o ConnectTimeout=10 drums-acceptance true 2>/dev/null && break; sleep 5; done
echo "$ID $AMI $IP $(date -u +%FT%TZ)"

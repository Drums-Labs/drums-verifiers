#!/usr/bin/env bash
# upgrade-drumsd.sh HOST DRUMSD_CHECKOUT: build drumsd at the checkout's HEAD on an installed host and switch to it
# (binary, OSWorld bridge, SDK wheel); restarts drumsd and drums-osw. Envs are destroyed by the restart.
set -euo pipefail
H=${1:?host}; SRC=${2:?drumsd checkout}
C=$(git -C "$SRC" rev-parse HEAD)
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "uncommitted changes in $SRC" >&2; exit 1; }
T=$(mktemp -d); git -C "$SRC" bundle create -q "$T/drumsd.bundle" HEAD
python3 "$SRC/install/build_wheel.py" "$T/sdk" > /dev/null
scp -q "$T/drumsd.bundle" "$H:/tmp/drumsd-up.bundle"; scp -q "$T"/sdk/drums-*.whl "$H:/tmp/"
ssh "$H" "set -e; sudo rm -rf /opt/drums/src/drumsd-up && sudo git clone -q /tmp/drumsd-up.bundle /opt/drums/src/drumsd-up && sudo git -C /opt/drums/src/drumsd-up checkout -q $C
  cd /opt/drums/src/drumsd-up && sudo env RUSTUP_HOME=/opt/drums/rust CARGO_HOME=/opt/drums/rust /opt/drums/rust/bin/cargo build -q --release --locked
  sudo systemctl stop drumsd drums-osw
  sudo install -m 0755 target/release/drumsd /opt/drums/bin/drumsd && sudo install -m 0644 osw/drums_osw.py /opt/drums/osw/drums_osw.py
  sudo rm -rf /opt/drums/src/drumsd && sudo mv /opt/drums/src/drumsd-up /opt/drums/src/drumsd
  sudo systemctl start drums-osw drumsd; sleep 3; systemctl is-active drumsd drums-osw
  for v in ~/drums-venv ~/t6/venv ~/ad-venv; do [ -x \$v/bin/pip ] && \$v/bin/pip -q install --force-reinstall --no-deps /tmp/drums-*.whl; done; true
  echo drumsd-upgraded-to $C"

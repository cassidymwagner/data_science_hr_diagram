## Commands to run docker properly

`docker build -t [image name] .`
`docker run --rm -it \
   --user=$(id -u) \
   --env="DISPLAY" \
   --workdir=/app \
   --volume="$PWD":/app \
   --volume="/etc/group:/etc/group:ro" \
   --volume="/etc/passwd:/etc/passwd:ro" \
   --volume="/etc/shadow:/etc/shadow:ro" \
   --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   [image name] python [py script]`

* X11 (and possibly other things) may be different when run on a Mac.

* `[image name]` list: 
  * `exoplanets_sandbox`
  * `agn_kde`
  * `stars`

* `[py script]` list:
  * `exoplanets_sandbox.py`
  * `agn_kde.py`
  * `stars.py`

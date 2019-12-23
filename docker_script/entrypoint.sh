#!/bin/bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback

DEF_NAME='user'
DEF_GROUP='usergroup'

USER_ID=${LOCAL_USER_ID:-9001}
USER_NAME=${LOCAL_USER_NAME:-$DEF_NAME}
GROUP_ID=${LOCAL_GROUP_ID:-1000}
GROUP_NAME=${LOCAL_GROUP_NAME:-$DEF_GROUP}

echo "Starting with UID : $USER_ID"
groupadd -g $GROUP_ID $GROUP_NAME
useradd --shell /bin/bash -u $USER_ID -g $GROUP_ID -o -c "" -m $USER_NAME
export HOME=/home/$USER_NAME

exec /usr/local/bin/gosu $USER_NAME "$@"

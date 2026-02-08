#!/bin/sh

set -xe
cc -Wall -Wextra -o nn nn.c 
./nn

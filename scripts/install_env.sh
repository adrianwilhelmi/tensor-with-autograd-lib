#!/bin/bash

function distro() {
	if [ -f /etc/debian_version ]; then
		PKG_UPDATE="apt-get update"
		PKG_INSTALL="apt-get install -y"
	elif [ -f /etc/redhat-release ]; then
		PKG_UPDATE="yum makecache -y"
		PKG_INSTALL="yum install -y"
	elif [ -f /etc/fedora-release ]; then
		PKG_UPDATE="dnf makecache -y"
		PKG_INSTALL="dnf install -y"
	elif [ -f /etc/arch-release ]; then
		PKG_UPDATE="pacman -Syu"
		PKG_INSTALL="pacman -S --noconfirm"
	elif [ -f /etc/os-release ]; then
		PKG_UPDATE="zypper refresh"
		PKG_INSTALL="zypper install -y"
	else 
		echo "Unknown distro"
		exit 1
	fi
		
	export PKG_UPDATE
	export PKG_INSTALL
}

distro

if ! sudo $PKG_UPDATE; then
	echo "update failed"
	exit 1
fi

if ! sudo $PKG_INSTALL make; then
	echo "installation failed"
	exit 1
fi

if ! sudo $PKG_INSTALL clang; then
	echo "installation failed"
	exit 1
fi

if ! sudo $PKG_INSTALL libopencv-dev; then
	echo "installation failed"
	exit 1
fi


echo "environment installed successfully."

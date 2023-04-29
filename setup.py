#!/usr/bin/env python

# Copyright (c) 2023, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/nr-catalog-tools/blob/master/LICENSE>

from __future__ import print_function

from os import environ, path
import subprocess
from pathlib import Path

NAME = "nrcatalogtools"
VERSION = "v0.0.1"


def write_version_file(version):
    """Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file (relative to the src package directory)
    """
    version_file = Path(NAME) / ".version"

    try:
        git_log = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%h %ai"]
        ).decode("utf-8")
        git_diff = (
            subprocess.check_output(["git", "diff", "."])
            + subprocess.check_output(["git", "diff", "--cached", "."])
        ).decode("utf-8")
    except subprocess.CalledProcessError as exc:  # git calls failed
        # we already have a version file, let's use it
        if version_file.is_file():
            return version_file.name
        # otherwise error out
        exc.args = (
            "unable to obtain git version information, and {} doesn't "
            "exist, cannot continue ({})".format(version_file, str(exc)),
        )
        raise
    else:
        git_version = "{}: ({}) {}".format(
            version, "UNCLEAN" if git_diff else "CLEAN", git_log.rstrip()
        )
        print("parsed git version info as: {!r}".format(git_version))

    try:
        with open(version_file, "w") as f:
            print(git_version, file=f)
            print("created {}".format(version_file))
    except:
        with open(str(version_file), "w") as f:
            print(git_version, file=f)
            print("created {}".format(version_file))

    return version_file.name


def get_long_description():
    """Finds the README and reads in the description"""
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, "README.md")) as f:
        long_description = f.read()
    return long_description


if "package_version" in environ:
    version = environ["package_version"]
    print("Setup.py using environment version='{0}'".format(version))
else:
    print("The variable 'package_version' was not present in the environment")
    try:
        from subprocess import check_output

        version = (
            check_output(
                """git log -1 --format=%cd --date=format:'%Y.%m.%d.%H.%M.%S'""",
                shell=use_shell,
            )
            .decode("ascii")
            .rstrip()
        )
        print("Setup.py using git log version='{0}'".format(version))
    except:
        from time import strftime, gmtime

        version = strftime("%Y.%m.%d.%H.%M.%S", gmtime())
        print("Setup.py using strftime version='{0}'".format(version))

if __name__ == "__main__":
    from setuptools import setup, find_packages

    setup(
        name=NAME,
        version=VERSION,
        description="A collection of tools to interface wtih Numerical Relativity waveform catalogs",
        long_description=get_long_description(),
        license="GPL",
        url="https://github.com/gwnrtools/nr-catalog-tools",
        author="Prayush Kumar",
        author_email="prayush.kumar@gmail.com",
        packages=find_packages(),
        package_dir={NAME: NAME},
        package_data={
            # version info
            NAME: [write_version_file(VERSION)],
            "template.data": [],
        },
        install_requires=[],
        scripts=[],
    )

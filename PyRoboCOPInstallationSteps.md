<!--
Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# PyRoboCOP Installation Steps:

In the following we show the steps for installating PyRoboCOP and its dependences. This installation requires that you have anaconda installed on your machine, and it is installed in the default location. If it is not installed in the default location, please be careful for step 5. In particular, we focus on the installation of pyadolc which is an external library to compute automatic differentiation in PyRoboCOP. The following installations have been tested on Mac OS (Mojave, Catalina and BigSur) and Ubuntu 16.04-18.04-20.04.

0. Access to PyRoboCOP main folder.
   ```
   cd PyRoboCOP
   ```

1. Create a python virtual environment with conda using the provided yml file:

   ```
   conda env create -f pyrobocop.yml
   ```

5. The above step will install the dependencies of pyrobocop as well as two other open-source optimization packages, casadi and pyomo, that we use for comparisons.

3. Activate the environment :

   ```
   conda activate pyrobocop
   ```



4. In a different location, clone the pyadolc branch compatible with python3 (https://github.com/b45ch1/pyadolc/tree/adolc-2.6.0_boost_1.66), be sure you are working on this branch:

   ```
    git clone -b adolc-2.6.0_boost_1.66 https://github.com/b45ch1/pyadolc.git
   ```



5. The following commands will be executed inside pyadolc
   ```
    cd pyadolc
    # Patch some of the pyadolc files (for standard Ubuntu OS)
    patch -p1 < ../pyadolc_install/pyadolc.patch
   ```

   and then do

   ```
   ./bootstrap.sh
   ```

   Note:

   a. on Mac if wget is not installed do:

   ```
   brew install wget
   ```

   b. on Mac if you have the following error " sh: aclocal: command not found", do the following:

   ```
   brew install automake
   ```

   c. after running ./bootstrap.sh if you have to run it again first do

   ```
   rm -R PACKAGES
   ```



6. Now set the boost folder environment variable:

   ```
   export BOOST_DIR=~/anaconda/envs/pyrobocop
   ```

   assuming anaconda is installed on the default location and you installed the conda env named "**pyrobocop**" in step 1. If your anaconda is installed elsewhere,

   ```
   'your_boost_path'=='/path_to_anaconda/envs/pyrobocop'
   ```

7. run

   ```
   python setup.py build
   python setup.py install
   ```

   If this succeeds, then you're done.

8. Check by  importing adolc in python.

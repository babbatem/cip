cd ~/installations

# get castxml
wget -q -O- https://data.kitware.com/api/v1/file/5e8b740d2660cbefba944189/download | tar zxf - -C ${HOME}
export PATH=${HOME}/castxml/bin:${PATH}

# install py things
python3 -m pip install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus

# download ompl 
OMPL="ompl"
git clone --recurse-submodules https://github.com/ompl/${OMPL}.git

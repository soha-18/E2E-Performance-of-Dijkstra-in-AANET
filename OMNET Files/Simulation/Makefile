###
# This file is responsible for providing the full toolchain of creating result graphs.
# Each target is responsible for one scenario, and should
# - start the simulation(s)
# - parse the resulting .sca and .vec files
# - and plot the parsed results into .pdfs
# 
# Before each target, some brief comments should describe what the scenario is about.
# Also, variables can be created that specify parameters of the respective target. These can be shared among several targets (where e.g. each target runs some sub-scenario).
# Naming these variables should follow a SCENARIO_VARNAME syntax.
###

update:
	cd ..; git pull

build-debug:	
	cd ../../; \
	echo -e "\ntracebasedapp"; \
	cd intairnet-tracebasedapp/src; opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET_dbg; make -j8 MODE=debug; cd ../..; \
	echo -e "\nTDMA"; \
	cd tdma/tdma/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../../inet4 -DINET_IMPORT -I../../../inet4/src -L../../../inet4/src -lINET_dbg; make -j8 MODE=debug; cd ../../..; \
	echo -e "\ndspr"; \
	cd dspr/src; opp_makemake -f --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I../../intairnet-tracebasedapp/src -I../../tdma/tdma/src -I. -I../../inet4/src -L../../inet4/src -L../../intairnet-tracebasedapp/out/gcc-debug/src/ -L../../tdma/tdma/out/gcc-debug/src/ -lINET_dbg -lintairnet-tracebasedapp_dbg -ltdma_dbg; make -j$(NUM_CPUS) MODE=debug; \

build-release:	
	cd ../../; \
	echo -e "\ntracebasedapp"; \
	cd intairnet-tracebasedapp/src; opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET; make -j8 MODE=release; cd ../..; \
	echo -e "\nTDMA"; \
	cd tdma/tdma/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../../inet4 -DINET_IMPORT -I../../../inet4/src -L../../../inet4/src -lINET; make -j8 MODE=release; cd ../../..; \
	echo -e "\ndspr"; \
	cd dspr/src; opp_makemake -f --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I../../intairnet-tracebasedapp/src -I../../tdma/tdma/src -I. -I../../inet4/src -L../../inet4/src -L../../intairnet-tracebasedapp/out/gcc-release/src/ -L../../tdma/tdma/out/gcc-release/src/ -lINET -lintairnet-tracebasedapp -ltdma; make -j4 MODE=release

cleanall:
	echo -e "\nclean dspr"; \
	cd ../src; make -j4 MODE=release clean; make -j4 MODE=debug clean; rm -f Makefile 


# GLOBAL VARIABLES  can be shared among all targets
# default binary in RELEASE mode
SIM_BINARY = ../out/gcc-release/src/dspr
SIM_BINARY_DBG = ../out/gcc-debug/src/dspr_dbg
# include .ned files
INCL = -n ../src:./:../../inet4/src/:../../tdma/tdma/src:../../intairnet-tracebasedapp/src

# END-to-END SIMUlATIONS

DSPR_TDMA_DEMO_CONFIGNAME = Twelve_nodes_scenario
DSPR_TDMA_NAC_CONFIGNAME = NAC_network
DSPR_TDMA_NAC_A2G_CONFIGNAME = NAC_network_modified

# run from dspr/simulations
dspr-tdma-demo:
	opp_runall -j4 $(SIM_BINARY) omnetpp.ini -c $(DSPR_TDMA_DEMO_CONFIGNAME) $(INCL)

dspr-tdma-demo-qtenv:
	opp_runall -j1 $(SIM_BINARY) omnetpp.ini -c $(DSPR_TDMA_DEMO_CONFIGNAME) $(INCL) -u Qtenv

dspr-tdma-demo-debug:
	gdbgui --args $(SIM_BINARY_DBG) omnetpp.ini -u Cmdenv -c $(DSPR_TDMA_DEMO_CONFIGNAME) $(INCL)

create-csv-12_node_scenario:
	cd results/Twelve_nodes_scenario/; \
	scavetool export -f 'module(Twelve_nodes_scenario.node[*].app[*]) AND name("packetSent:count")' -o packet_sent.csv sendInterval\=*.sca; \
	scavetool export -f 'module(Twelve_nodes_scenario.node[12].app[*]) AND name("packetReceived:count")' -o packet_received.csv sendInterval\=*.sca; \
	scavetool export -f 'module(Twelve_nodes_scenario.node[12].app[*]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv sendInterval\=*.vec; \
	scavetool export -f 'module(Twelve_nodes_scenario.node[*].routing) AND name("packetIdSent:vector")' -o packetId_sent_vector.csv sendInterval\=*.vec; \
	scavetool export -f 'module(Twelve_nodes_scenario.node[12].routing) AND name("packetIDReceived:vector")' -o packetId_received_vector.csv sendInterval\=*.vec; \
	scavetool export -f 'module(Twelve_nodes_scenario.node[12].routing) AND name("hopCount:vector")' -o hopCount_vector.csv sendInterval\=*.vec; \
	scavetool export -f 'module(Twelve_nodes_scenario.node[*].wlan[*].mac.queue) AND name("queueingTime:vector")' -o queueingTime_vector.csv sendInterval\=*.vec; \
	scavetool export -f 'module(Twelve_nodes_scenario.node[*].wlan[*].mac.queue) AND name("queueLength:vector")' -o queueLength_vector.csv sendInterval\=*.vec;



dspr-tdma-nac-demo:
	opp_runall -j4 $(SIM_BINARY) NAC_network.ini -c $(DSPR_TDMA_NAC_CONFIGNAME) $(INCL)

dspr-tdma-nac-demo-qtenv:
	opp_runall -j1 $(SIM_BINARY) NAC_network.ini -c $(DSPR_TDMA_NAC_CONFIGNAME) $(INCL) -u Qtenv

dspr-tdma-nac-demo-debug:
	gdbgui --args $(SIM_BINARY_DBG) NAC_network.ini -u Cmdenv -c $(DSPR_TDMA_NAC_CONFIGNAME) $(INCL)

create-csv-nac_scenario:
	cd results/NAC_network/; \
	scavetool export -f 'module(NAC_network.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent.csv sendInterval\=*.sca; \
	scavetool export -f 'module(NAC_network.groundStation[0].app[*]) AND name("packetReceived:count")' -o packet_received.csv sendInterval\=*.sca; \
	scavetool export -f 'module(NAC_network.groundStation[0].app[*]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv sendInterval\=*.vec; \
	scavetool export -f 'module(NAC_network.aircraft[*].routing) AND name("packetIdSent:vector")' -o packetId_sent_vector.csv sendInterval\=*.vec; \
	scavetool export -f 'module(NAC_network.groundStation[0].routing) AND name("packetIDReceived:vector")' -o packetId_received_vector.csv sendInterval\=*.vec; \
	scavetool export -f 'module(NAC_network.groundStation[0].routing) AND name("hopCount:vector")' -o hopCount_vector.csv sendInterval\=*.vec; \
	scavetool export -f 'module(NAC_network.aircraft[*].wlan[*].mac.queue) AND name("queueingTime:vector")' -o queueingTime_vector.csv sendInterval\=*.vec; \
	scavetool export -f 'module(NAC_network.aircraft[*].wlan[*].mac.queue) AND name("queueLength:vector")' -o queueLength_vector.csv sendInterval\=*.vec;



dspr-tdma-nac-a2g-demo:
	opp_runall -j4 $(SIM_BINARY) NAC_network.ini -c $(DSPR_TDMA_NAC_A2G_CONFIGNAME) $(INCL)

dspr-tdma-nac-a2g-demo-qtenv:
	opp_runall -j1 $(SIM_BINARY) NAC_network.ini -c $(DSPR_TDMA_NAC_A2G_CONFIGNAME) $(INCL) -u Qtenv

dspr-tdma-nac-a2g-demo-debug:
	gdbgui --args $(SIM_BINARY_DBG) NAC_network.ini -u Cmdenv -c $(DSPR_TDMA_NAC_A2G_CONFIGNAME) $(INCL)

create-csv-nac_a2g_scenario:
	cd results/; \
	#scavetool export -f 'module(NAC_network_modified.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent.csv sendInterval\=*.sca; \
	#scavetool export -f 'module(NAC_network_modified.groundStation[0].app[*]) AND name("packetReceived:count")' -o packet_received.csv sendInterval\=*.sca; \
	#scavetool export -f 'module(NAC_network_modified.groundStation[0].app[*]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv sendInterval\=*.vec; \
	#scavetool export -f 'module(NAC_network_modified.aircraft[*].routing) AND name("packetIdSent:vector")' -o packetId_sent_vector.csv sendInterval\=*.vec; \
	#scavetool export -f 'module(NAC_network_modified.groundStation[0].routing) AND name("packetIDReceived:vector")' -o packetId_received_vector.csv sendInterval\=*.vec; \
	#scavetool export -f 'module(NAC_network_modified.groundStation[0].routing) AND name("hopCount:vector")' -o hopCount_vector.csv sendInterval\=*.vec; \
	#scavetool export -f 'module(NAC_network_modified.aircraft[*].wlan[*].mac.queue) AND name("queueingTime:vector")' -o queueingTime_vector.csv sendInterval\=*.vec; \
	#scavetool export -f 'module(NAC_network_modified.aircraft[*].wlan[*].mac.queue) AND name("queueLength:vector")' -o queueLength_vector.csv sendInterval\=*.vec;\
	scavetool export -f 'module(NAC_network_modified.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent.csv commRange=*.sca; \
	scavetool export -f 'module(NAC_network_modified.groundStation[0].app[*]) AND name("packetReceived:count")' -o packet_received.csv commRange=*.sca; \
	scavetool export -f 'module(NAC_network_modified.groundStation[0].app[*]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv commRange\=*.vec; \
	scavetool export -f 'module(NAC_network_modified.aircraft[*].routing) AND name("packetIdSent:vector")' -o packetId_sent_vector.csv commRange\=*.vec; \
	scavetool export -f 'module(NAC_network_modified.groundStation[0].routing) AND name("packetIDReceived:vector")' -o packetId_received_vector.csv commRange\=*.vec; \
	scavetool export -f 'module(NAC_network_modified.groundStation[0].routing) AND name("hopCount:vector")' -o hopCount_vector.csv commRange\=*.vec; \
	scavetool export -f 'module(NAC_network_modified.aircraft[*].wlan[*].mac.queue) AND name("queueingTime:vector")' -o queueingTime_vector.csv commRange\=*.vec; \
	scavetool export -f 'module(NAC_network_modified.aircraft[*].wlan[*].mac.queue) AND name("queueLength:vector")' -o queueLength_vector.csv commRange\=*.vec;


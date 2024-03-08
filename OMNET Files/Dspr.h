/*
 * Dspr.h
 *
 *  Created on: Aug 30, 2023
 *      Author: sohini
 */

#ifndef DSPR_H_
#define DSPR_H_

#include <omnetpp.h>
#include <vector>
#include <map>
#include "inet/common/INETDefs.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/geometry/common/Coord.h"
#include "inet/common/packet/chunk/Chunk.h"
#include "inet/networklayer/common/L3Address.h"
#include "inet/networklayer/contract/IL3AddressType.h"
#include "inet/networklayer/contract/INetfilter.h"
#include "inet/networklayer/contract/IRoutingTable.h"
#include "inet/linklayer/common/InterfaceTag_m.h"
#include "inet/routing/base/RoutingProtocolBase.h"
#include "inet/networklayer/contract/IInterfaceTable.h"
#include "inet/transportlayer/udp/UdpHeader_m.h"
#include "inet/networklayer/ipv4/Ipv4InterfaceData.h"
#include "NodeManager.h"
#include "Dspr_m.h"
#include "DsprDefs.h"

using namespace omnetpp;
using namespace inet;

class NodeManager;



class Dspr : public RoutingProtocolBase, public cListener, public NetfilterBase::HookBase
{
  private:
    NodeManager* nodeManager = nullptr;
    // cLongHistogram hopCountStats;
	  // cOutVector hopCountVector;
    simsignal_t packetIdSentSignal;
    simsignal_t packetIDReceivedSignal;

    int packetID = 0;

  public:
    cModule *node = nullptr;
    IMobility *mobility = nullptr;
    int interfaceId = -1;
    int packetReceived = 0;
    bool displayBubbles;
    double groundStationRange;

   
    IInterfaceTable *interfaceTable = nullptr;
    IRoutingTable *routingTable = nullptr;
    const char *outputInterface = nullptr;
    InterfaceEntry *interfaceptr = nullptr;
    IL3AddressType *addressType = nullptr;
    INetfilter *networkProtocol = nullptr;
    const char *interfaces = nullptr;

    //Enabling A2G interface if nexthop is GS
    const char *a2gOutputInterface = nullptr;

    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
    void initialize(int stage); // override;
    void handleMessageWhenUp(cMessage *msg) override;
    void handleMessageWhenDown(cMessage *msg) override;
    void printInterfaceTable();
    // void processSelfMessage(cMessage *msg);
    void processMessage(cMessage *msg);
    // configuration
    void configureInterfaces();

    L3Address getSelfAddress() const;
    L3Address getSenderNeighborAddress(const Ptr<const NetworkHeaderBase>& networkHeader) const;


   // handling packets
     DsprInfo *createDsprInfo();
     void setDsprInfoOnNetworkDatagram(Packet *packet, const Ptr<const NetworkHeaderBase>& networkHeader, DsprInfo *dsprInfo);

     // returns nullptr if not found
     DsprInfo *findDsprInfoInNetworkDatagramForUpdate(const Ptr<NetworkHeaderBase>& networkHeader);
     const DsprInfo *findDsprInfoInNetworkDatagram(const Ptr<const NetworkHeaderBase>& networkHeader) const;

     // throws an error when not found
     DsprInfo *getDsprInfoFromNetworkDatagramForUpdate(const Ptr<NetworkHeaderBase>& networkHeader);
     const DsprInfo *getDsprInfoFromNetworkDatagram(const Ptr<const NetworkHeaderBase>& networkHeader) const;


    Coord lookupPositionInGlobalRegistry(const L3Address& address) const;
    void storePositionInGlobalRegistry(const L3Address& address, const Coord& position) const;
    void storeSelfPositionInGlobalRegistry() const;

    // routing
    Result routeDatagram(Packet *datagram, DsprInfo *dsprInfo);

    // netfilter
    virtual Result datagramPreRoutingHook(Packet *datagram) override; // { return ACCEPT; };
    virtual Result datagramForwardHook(Packet *datagram) override { return ACCEPT; }
    virtual Result datagramPostRoutingHook(Packet *datagram) override { return ACCEPT; }
    virtual Result datagramLocalInHook(Packet *datagram) override; // { return ACCEPT; }
    virtual Result datagramLocalOutHook(Packet *datagram) override; //{ return ACCEPT; };



    // lifecycle
    virtual void handleStartOperation(LifecycleOperation *operation) override;
    virtual void handleStopOperation(LifecycleOperation *operation) override;
    virtual void handleCrashOperation(LifecycleOperation *operation) override;


    simsignal_t hopCountSignal;
    // notification
    virtual void receiveSignal(cComponent *source, simsignal_t signalID, cObject *obj, cObject *details) override;
    //void finish();
};


#endif /* DSPR_H_ */

/*
 * Dspr.cc
 *
 *  Created on: Aug 30, 2023
 *      Author: sohini
 */
#include <algorithm>
#include <typeinfo>
#include<fstream>
#include <string>
#include "Dspr.h"
#include "omnetpp.h"
#include "inet/common/INETUtils.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/IProtocolRegistrationListener.h"
#include "inet/common/ProtocolTag_m.h"
#include "inet/linklayer/common/InterfaceTag_m.h"
#include "inet/common/lifecycle/ModuleOperations.h"
#include "inet/networklayer/common/HopLimitTag_m.h"
#include "inet/networklayer/common/IpProtocolId_m.h"
#include "inet/networklayer/common/L3AddressTag_m.h"
#include "inet/networklayer/common/L3Tools.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/networklayer/common/NextHopAddressTag_m.h"

#define NODE_SHUTDOWN_EVENT 1

#ifdef WITH_IPv4
#include "inet/networklayer/ipv4/Ipv4Header_m.h"
#endif

#ifdef WITH_IPv6
#include "inet/networklayer/ipv6/Ipv6ExtensionHeaders_m.h"
#include "inet/networklayer/ipv6/Ipv6InterfaceData.h"
#endif

#ifdef WITH_NEXTHOP
#include "inet/networklayer/nexthop/NextHopForwardingHeader_m.h"
#endif

using namespace inet;

Define_Module(Dspr);


void Dspr::initialize(int stage)
{
    if (stage == INITSTAGE_ROUTING_PROTOCOLS)
       addressType = getSelfAddress().getAddressType();


    RoutingProtocolBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL){
      nodeManager = dynamic_cast<NodeManager *>(getModuleByPath("^.^.nodeManager"));
      node = getContainingNode(this);
      mobility = check_and_cast<IMobility *>(node->getSubmodule("mobility"));
      interfaceTable = getModuleFromPar<IInterfaceTable>(par("interfaceTableModule"), this);
      interfaceptr = interfaceTable->findInterfaceByName("wlan0");
      interfaces = par("interfaces");
      routingTable = getModuleFromPar<IRoutingTable>(par("routingTableModule"), this);
      outputInterface = par("outputInterface");
      int nodeId = node->getIndex();
      EV<<"Node ID: " << nodeId << endl;
      nodeManager->registerClient(node);
      displayBubbles = par("displayBubbles");
      //scheduleAt(1800, new cMessage("NodeShutdownEvent", NODE_SHUTDOWN_EVENT));
      networkProtocol = getModuleFromPar<INetfilter>(par("networkProtocolModule"), this);
      hopCountSignal = registerSignal("hopCount");
    //   hopCountStats.setName("hopCountStats");
    //   hopCountVector.setName("HopCount");
      packetIdSentSignal = registerSignal("packetIdSent");
      packetIDReceivedSignal = registerSignal("packetIDReceived");

      //const char *file_name = par("groundstationsTraceFile");
      //parseGroundstationTraceFile2Vector(file_name); //working
      a2gOutputInterface = par("a2gOutputInterface");
      groundStationRange = par("groundStationRange");
        }
    else if (stage == INITSTAGE_ROUTING_PROTOCOLS) {
        registerService(Protocol::manet, nullptr, gate("ipIn"));
        registerProtocol(Protocol::manet, gate("ipOut"), nullptr);
        node->subscribe(linkBrokenSignal, this);
        networkProtocol->registerHook(0, this);
       }
  }

void Dspr::configureInterfaces()
{
    cPatternMatcher interfaceMatcher(interfaces, false, true, false);
    int numInterfaces = interfaceTable->getNumInterfaces();
    EV << "Number of Interfaces: " << numInterfaces << endl;

    for (int i = 0; i < numInterfaces; i++) {
        interfaceptr = interfaceTable->getInterface(i);
        if (interfaceptr && interfaceMatcher.matches(interfaceptr->getInterfaceName())) {
            if (interfaceptr->isMulticast()) {
                interfaceptr->joinMulticastGroup(addressType->getLinkLocalManetRoutersMulticastAddress());
                EV << "Configured multicast for interface: " << interfaceptr->getInterfaceName() << endl; //working

            }
        }
    }
}

void Dspr::printInterfaceTable()
{
    if (interfaceTable) {
      EV << "Printing Interface Table:" << endl;
      int numInterfaces = interfaceTable->getNumInterfaces();
      for (int i = 0; i < numInterfaces; i++) {
          interfaceptr = interfaceTable->getInterface(i);
          EV << "Interface " << i << ":" << endl;
          EV << "  Name: " << interfaceptr->getInterfaceName() << endl;
          cModule *parentModule = interfaceptr->getParentModule();
          int nodeID = parentModule->getIndex();
          EV << "  Node ID: " << nodeID << endl;
          //EV << "  Node ID: " << interfaceptr->getIndex()<< endl; //getNodeId() << endl;
          EV << "  IP Address: " << interfaceptr->getProtocolData<Ipv4InterfaceData>()->getIPAddress() << endl;
       }
   } else {
      EV << "InterfaceTable is not available." << endl;
   }
}

void Dspr::handleMessageWhenUp(cMessage *msg){
   if (msg->isSelfMessage())
       delete msg;
       //processSelfMessage(msg);

    else
       processMessage(msg);
}

void Dspr::handleMessageWhenDown(cMessage *msg)
{
    if (msg->isSelfMessage()) {
       nodeManager->deregisterClient(node);
       delete msg;
    }
    else
        throw cRuntimeError("Unknown message");
    
}
// void Dspr::processSelfMessage(cMessage *msg)
// {
//     if (msg->getKind() == NODE_SHUTDOWN_EVENT){
//        nodeManager->deregisterClient(node);
//         delete msg;
//     }
// }


void Dspr::processMessage(cMessage *msg){
    //  simtime_t eed = simTime() - msg->getCreationTime();
    //  cModule* sender = msg->getSenderModule();
    //  EV << "Delay: " << eed << "from node: "<<  sender->getIndex()  <<endl;
     if (auto packet = dynamic_cast<Packet *>(msg)){
        packetReceived++;
        delete msg;
       }
     else
        throw cRuntimeError("Unknown message");
        //EV << "Received an unknown message type: " << msg->getClassName() << endl;

 }

L3Address Dspr::getSelfAddress() const
{
    //TODO choose self address based on a new 'interfaces' parameter
L3Address ret = routingTable->getRouterIdAsGeneric();
#ifdef WITH_IPv6
    if (ret.getType() == L3Address::IPv6) {
        for (int i = 0; i < interfaceTable->getNumInterfaces(); i++) {
            InterfaceEntry *ie = interfaceTable->getInterface(i);
            if ((!ie->isLoopback())) {
                if (auto ipv6Data = ie->findProtocolData<Ipv6InterfaceData>()) {
                    ret = ipv6Data->getPreferredAddress();
                    break;
                }
            }
        }
    }
#endif
    return ret;
}

L3Address Dspr::getSenderNeighborAddress(const Ptr<const NetworkHeaderBase>& networkHeader) const
{
    const DsprInfo *dsprInfo = getDsprInfoFromNetworkDatagram(networkHeader);
    return dsprInfo->getSenderAddress();
}

DsprInfo *Dspr::createDsprInfo()
{
    DsprInfo *dsprInfo = new DsprInfo();
    int addressesBytes = 3 * getSelfAddress().getAddressType()->getAddressByteLength();
    int tlBytes = 1 + 1;
    int packetLength = tlBytes + addressesBytes;
    dsprInfo->setLength(packetLength);
    return dsprInfo;
}


//
// routing
//

INetfilter::IHook::Result Dspr::routeDatagram(Packet *datagram, DsprInfo *dsprInfo)
{
    const auto& networkHeader = getNetworkProtocolHeader(datagram);
    const L3Address& source = getSelfAddress(); 
    const L3Address& destination = networkHeader->getDestinationAddress();
    EV_INFO << "Finding next hop: source = " << source << ", destination = " << destination << endl;
    L3Address nextHopAddress = nodeManager->findNextHop(source, destination);
    datagram->addTagIfAbsent<NextHopAddressReq>()->setNextHopAddress(nextHopAddress);
    if (nextHopAddress.isUnspecified()) {
        EV_WARN << "No next hop found, dropping packet: source = " << source << ", destination = " << destination << endl;
        if (displayBubbles && hasGUI())
            getContainingNode(node)->bubble("No next hop found, dropping packet");
        return DROP;
    }
    else {
        EV_INFO << "Next hop found: source = " << source << ", destination = " << destination << ", nextHop: " << nextHopAddress << endl;
        dsprInfo->setSenderAddress(source);
        dsprInfo->setCurrentSenderAddress(getSelfAddress());
        dsprInfo->setCurrentReceiverAddress(nextHopAddress);
        // auto interfaceEntry = CHK(interfaceTable->findInterfaceByName(outputInterface));
        // datagram->addTagIfAbsent<InterfaceReq>()->setInterfaceId(interfaceEntry->getInterfaceId());
        // L3Address groundStationAddress = nodeManager->destAddress;
        Coord destination_position = nodeManager->destPosition;
        EV << "Destination Position is: " << destination_position << endl;
        Coord current_aircraft_position = mobility->getCurrentPosition();
        double distanceToGroundStation = current_aircraft_position.distance(destination_position);
        if(distanceToGroundStation <= groundStationRange){
        /////A2G link implementation///////
        // if (nextHopAddress == groundStationAddress) {
           EV << " Ground Station is within communication range." << endl;
           EV << "Switch on the Air-to-Ground Link." << endl;
           auto a2gInterfaceEntry = CHK(interfaceTable->findInterfaceByName(a2gOutputInterface));
           datagram->addTagIfAbsent<InterfaceReq>()->setInterfaceId(a2gInterfaceEntry->getInterfaceId());
           EV_INFO << " Interface ID = " << a2gInterfaceEntry << endl;
           return ACCEPT;
        }
        EV << "Transmit through Air-to-Air link." << endl;
        auto interfaceEntry = CHK(interfaceTable->findInterfaceByName(outputInterface));
        datagram->addTagIfAbsent<InterfaceReq>()->setInterfaceId(interfaceEntry->getInterfaceId());
        EV_INFO << " Interface ID = " << interfaceEntry << endl;
        return ACCEPT;
    }
}

void Dspr::setDsprInfoOnNetworkDatagram(Packet *packet, const Ptr<const NetworkHeaderBase>& networkHeader, DsprInfo *dsprInfo)
{
    packet->trimFront();
#ifdef WITH_IPv4
    if (dynamicPtrCast<const Ipv4Header>(networkHeader)) {
        auto ipv4Header = removeNetworkProtocolHeader<Ipv4Header>(packet);
        dsprInfo->setType(IPOPTION_TLV_GPSR);
        B oldHlen = ipv4Header->calculateHeaderByteLength();
        ASSERT(ipv4Header->getHeaderLength() == oldHlen);
        ipv4Header->addOption(dsprInfo);
        B newHlen = ipv4Header->calculateHeaderByteLength();
        ipv4Header->setHeaderLength(newHlen);
        ipv4Header->addChunkLength(newHlen - oldHlen);
        ipv4Header->setTotalLengthField(ipv4Header->getTotalLengthField() + newHlen - oldHlen);
        insertNetworkProtocolHeader(packet, Protocol::ipv4, ipv4Header);
    }
    else
#endif
#ifdef WITH_IPv6
    if (dynamicPtrCast<const Ipv6Header>(networkHeader)) {
        auto ipv6Header = removeNetworkProtocolHeader<Ipv6Header>(packet);
        dsprInfo->setType(IPv6TLVOPTION_TLV_GPSR);
        B oldHlen = ipv6Header->calculateHeaderByteLength();
        Ipv6HopByHopOptionsHeader *hdr = check_and_cast_nullable<Ipv6HopByHopOptionsHeader *>(ipv6Header->findExtensionHeaderByTypeForUpdate(IP_PROT_IPv6EXT_HOP));
        if (hdr == nullptr) {
            hdr = new Ipv6HopByHopOptionsHeader();
            hdr->setByteLength(B(8));
            ipv6Header->addExtensionHeader(hdr);
        }
        hdr->getTlvOptionsForUpdate().insertTlvOption(dsprInfo);
        hdr->setByteLength(B(utils::roundUp(2 + B(hdr->getTlvOptions().getLength()).get(), 8)));
        B newHlen = ipv6Header->calculateHeaderByteLength();
        ipv6Header->addChunkLength(newHlen - oldHlen);
        insertNetworkProtocolHeader(packet, Protocol::ipv6, ipv6Header);
    }
    else
#endif
#ifdef WITH_NEXTHOP
    if (dynamicPtrCast<const NextHopForwardingHeader>(networkHeader)) {
        auto nextHopHeader = removeNetworkProtocolHeader<NextHopForwardingHeader>(packet);
        dsprInfo->setType(NEXTHOP_TLVOPTION_TLV_GPSR);
        int oldHlen = nextHopHeader->getTlvOptions().getLength();
        nextHopHeader->getTlvOptionsForUpdate().insertTlvOption(dsprInfo);
        int newHlen = nextHopHeader->getTlvOptions().getLength();
        nextHopHeader->addChunkLength(B(newHlen - oldHlen));
        insertNetworkProtocolHeader(packet, Protocol::nextHopForwarding, nextHopHeader);
    }
    else
#endif
    {
    }
}

const DsprInfo *Dspr::findDsprInfoInNetworkDatagram(const Ptr<const NetworkHeaderBase>& networkHeader) const
{
    const DsprInfo *dsprInfo = nullptr;

#ifdef WITH_IPv4
    if (auto ipv4Header = dynamicPtrCast<const Ipv4Header>(networkHeader)) {
        dsprInfo = check_and_cast_nullable<const DsprInfo *>(ipv4Header->findOptionByType(IPOPTION_TLV_GPSR));
    }
    else
#endif
#ifdef WITH_IPv6
    if (auto ipv6Header = dynamicPtrCast<const Ipv6Header>(networkHeader)) {
        const Ipv6HopByHopOptionsHeader *hdr = check_and_cast_nullable<const Ipv6HopByHopOptionsHeader *>(ipv6Header->findExtensionHeaderByType(IP_PROT_IPv6EXT_HOP));
        if (hdr != nullptr) {
            int i = (hdr->getTlvOptions().findByType(IPv6TLVOPTION_TLV_GPSR));
            if (i >= 0)
                dsprInfo = check_and_cast<const DsprInfo *>(hdr->getTlvOptions().getTlvOption(i));
        }
    }
    else
#endif
#ifdef WITH_NEXTHOP
    if (auto nextHopHeader = dynamicPtrCast<const NextHopForwardingHeader>(networkHeader)) {
        int i = (nextHopHeader->getTlvOptions().findByType(NEXTHOP_TLVOPTION_TLV_GPSR));
        if (i >= 0)
            dsprInfo = check_and_cast<const DsprInfo *>(nextHopHeader->getTlvOptions().getTlvOption(i));
    }
    else
#endif
    {
    }
    return dsprInfo;
}

DsprInfo *Dspr::findDsprInfoInNetworkDatagramForUpdate(const Ptr<NetworkHeaderBase>& networkHeader)
{
    DsprInfo *dsprInfo = nullptr;

#ifdef WITH_IPv4
    if (auto ipv4Header = dynamicPtrCast<Ipv4Header>(networkHeader)) {
        dsprInfo = check_and_cast_nullable<DsprInfo *>(ipv4Header->findMutableOptionByType(IPOPTION_TLV_GPSR));
    }
    else
#endif
#ifdef WITH_IPv6
    if (auto ipv6Header = dynamicPtrCast<Ipv6Header>(networkHeader)) {
        Ipv6HopByHopOptionsHeader *hdr = check_and_cast_nullable<Ipv6HopByHopOptionsHeader *>(ipv6Header->findExtensionHeaderByTypeForUpdate(IP_PROT_IPv6EXT_HOP));
        if (hdr != nullptr) {
            int i = (hdr->getTlvOptions().findByType(IPv6TLVOPTION_TLV_GPSR));
            if (i >= 0)
                dsprInfo = check_and_cast<DsprInfo *>(hdr->getTlvOptionsForUpdate().getTlvOptionForUpdate(i));
        }
    }
    else
#endif
#ifdef WITH_NEXTHOP
    if (auto nextHopHeader = dynamicPtrCast<NextHopForwardingHeader>(networkHeader)) {
        int i = (nextHopHeader->getTlvOptions().findByType(NEXTHOP_TLVOPTION_TLV_GPSR));
        if (i >= 0)
            dsprInfo = check_and_cast<DsprInfo *>(nextHopHeader->getTlvOptionsForUpdate().getTlvOptionForUpdate(i));
    }
    else
#endif
    {
    }
    return dsprInfo;
}

const DsprInfo *Dspr::getDsprInfoFromNetworkDatagram(const Ptr<const NetworkHeaderBase>& networkHeader) const
{
    const DsprInfo *dsprInfo = findDsprInfoInNetworkDatagram(networkHeader);
    if (dsprInfo == nullptr)
        throw cRuntimeError("Dspr info not found in datagram!");
    return dsprInfo;
}

DsprInfo *Dspr::getDsprInfoFromNetworkDatagramForUpdate(const Ptr<NetworkHeaderBase>& networkHeader)
{
    DsprInfo *dsprInfo = findDsprInfoInNetworkDatagramForUpdate(networkHeader);
    if (dsprInfo == nullptr)
        throw cRuntimeError("Dspr info not found in datagram!");
    return dsprInfo;
}

//
// netfilter
//

INetfilter::IHook::Result Dspr::datagramPreRoutingHook(Packet *datagram)
{
    EV_INFO << "I am datagramPreRoutingHook " << endl;   
    Enter_Method("datagramPreRoutingHook");
    const auto& networkHeader = getNetworkProtocolHeader(datagram);
    const L3Address& destination = networkHeader->getDestinationAddress();
    if (destination.isMulticast() || destination.isBroadcast() || routingTable->isLocalAddress(destination))
        return ACCEPT;
    else {
        // KLUDGE: this allows overwriting the Dspr option inside
        auto dsprInfo = const_cast<DsprInfo *>(getDsprInfoFromNetworkDatagram(networkHeader));
        return routeDatagram(datagram, dsprInfo);
    }
}

INetfilter::IHook::Result Dspr::datagramLocalOutHook(Packet *packet)
{
    Enter_Method("datagramLocalOutHook");
    EV_INFO << "I am datagramLocalOutHook " << endl;

    double combinedId = node->getIndex() + static_cast<double>(packetID) / 10000.0;
    //EV_INFO << "Sohini" << combinedId << endl;
    emit(packetIdSentSignal, combinedId);
    packetID += 1;
    const auto& networkHeader = getNetworkProtocolHeader(packet);
    const L3Address& destination = networkHeader->getDestinationAddress();
    if (destination.isMulticast() || destination.isBroadcast() || routingTable->isLocalAddress(destination))
        return ACCEPT;
    else {
        DsprInfo *dsprInfo = createDsprInfo();
        dsprInfo->setNodeIndex(combinedId);
        setDsprInfoOnNetworkDatagram(packet, networkHeader, dsprInfo);
        return routeDatagram(packet, dsprInfo);
    }
    
}

INetfilter::IHook::Result Dspr::datagramLocalInHook(Packet *packet)
{
    Enter_Method("datagramLocalInHook");
    EV_INFO << "I am datagramLocalInHook " << endl;
    const auto& ipv4Header = packet->peekAtFront<Ipv4Header>();
    const auto& networkHeader = getNetworkProtocolHeader(packet);
    const L3Address& destination = networkHeader->getDestinationAddress();
    const L3Address& source = networkHeader->getDestinationAddress();
    const DsprInfo *dsprInfo = findDsprInfoInNetworkDatagram(networkHeader);
    if (dsprInfo != nullptr){
        EV_INFO << "Packet Received " << endl;
        double receivedId = dsprInfo->getNodeIndex();
        emit(packetIDReceivedSignal, receivedId);
        emit(hopCountSignal, 32 - (ipv4Header->getTimeToLive()) + 1);
    }
    return ACCEPT;
}

void Dspr::handleStartOperation(LifecycleOperation *operation)
{
    configureInterfaces();
}

void Dspr::handleStopOperation(LifecycleOperation *operation)
{
   nodeManager->deregisterClient(node);
   EV << "Total packets received at the destination node: " << packetReceived << endl;
}

void Dspr::handleCrashOperation(LifecycleOperation *operation)
{
   nodeManager->deregisterClient(node);
   EV << "Total packets received at the destination node: " << packetReceived << endl;
}

// void Dspr::finish(){

//     EV << "Hop count, min:    " << hopCountStats.getMin() << endl;
// 	EV << "Hop count, max:    " << hopCountStats.getMax() << endl;
// 	EV << "Hop count, mean:   " << hopCountStats.getMean() << endl;
// 	EV << "Hop count, stddev: " << hopCountStats.getStddev() << endl;

//     hopCountStats.recordAs("hop count");
// }

void Dspr::receiveSignal(cComponent *source, simsignal_t signalID, cObject *obj, cObject *details)
{
    Enter_Method("receiveChangeNotification");
    if (signalID == linkBrokenSignal) {
        EV_WARN << "Received link break" << endl;
        // TODO: remove the neighbor
    }
}

//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

package dspr;

import inet.routing.contract.IManetRouting;


simple Dspr like IManetRouting
{
    parameters:
        @display("i=block/routing");
        //int numNodes;
        string routingTableModule = default("^.ipv4.routingTable");
        string interfaceTableModule;
        string interfaces = default("*");
        string outputInterface = default("wlan0");
        string a2gOutputInterface = default("eth0");
        string networkProtocolModule = default("^.ipv4.ip");
        
        double groundStationRange @unit(m) = default(370400m); 
        //string groundstationsTraceFile = default("groundstations.txt");      
        bool displayBubbles = default(false);
        
        @signal[hopCount](type=long);
        @statistic[hopCount](source=hopCount; record=vector);

        @signal[packetIdSent](type=double);
        @statistic[packetIdSent](source=packetIdSent; record=vector);

        @signal[packetIDReceived](type=double);
        @statistic[packetIDReceived](source=packetIDReceived; record=vector);

        
    gates:
        input ipIn;
        output ipOut;

}

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
//import inet.node.inet.INetworkNode;
package dspr.simulations;

import inet.physicallayer.contract.packetlevel.IRadioMedium;
import inet.node.inet.INetworkNode;
import inet.common.scenario.ScenarioManager;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.ManetRouter;
import inet.visualizer.contract.IIntegratedVisualizer;
import tdma.scheduler.TdmaScheduler;
import dspr.NodeManager;


network Twelve_nodes_scenario
{
    parameters:
        double commRange = 0;
        int numNodes = default(0); //default(0); // = 10;
        @display("bgb=882,562");
        @figure[title](type=label; pos=0,-1; anchor=sw; color=darkblue);

    submodules:
        scenarioManager: ScenarioManager;
        visualizer: <default("IntegratedCanvasVisualizer")> like IIntegratedVisualizer if hasVisualizer() {
            @display("p=809,124");
        }
        radioMedium: <default("UnitDiskRadioMedium")> like IRadioMedium {
            @display("p=817,306");
        }
        
        configurator: Ipv4NetworkConfigurator {
            @display("p=817,217");
        }
       //scheduler: TdmaScheduler {
            //parameters:
                //@display("p=100,400;is=s");
        //}

        node[numNodes]: ManetRouter; 
        
        //groundStation: ManetRouter {
            //parameters:
                //@display("p=250,125");
        //}
        
        nodeManager: NodeManager {
           @display("p=796,55");
        }

    connections allowunconnected:
}


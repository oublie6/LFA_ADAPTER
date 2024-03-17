/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

const bit<8>  HULAPP_PROTOCOL = 254; 
const bit<8>  HULAPP_DATA_PROTOCOL = 253;
const bit<8>  HULAPP_BACKGROUND_PROTOCOL = 252;
const bit<8>  HULAPP_TCP_DATA_PROTOCOL = 251;
const bit<8>  HULAPP_UDP_DATA_PROTOCOL = 250;
const bit<8>  TCP_PROTOCOL = 6;
const bit<8>  UDP_PROTOCOL = 17;
const bit<16> TYPE_IPV4 = 0x0800;
const bit<16> TYPE_ARP  = 0x0806;
const bit<9>  LOOP_THRESHOLD = 3;
const bit<48> FLOWLET_TIMEOUT = 200000;
const bit<48> LINK_TIMEOUT = 800000;
//#define TAU_EXPONENT 9 // twice the probe frequency. if probe freq = 256 microsec, the TAU should be 512 microsec, and the TAU_EXPONENT would be 9
#define TAU_EXPONENT 19 // twice the probe frequency. if probe freq = 256 millisec, the TAU should be 512 millisec, and the TAU_EXPONENT would be 19
const bit<32> UTIL_RESET_TIME_THRESHOLD = 512000000;

register<bit<32>>(2048) min_path_util; //Min path util on a port
register<bit<9>>(2048) best_nhop;     //Best next hop
register<bit<32>>(2048) version;     //Link util on a port.

// Metric util
register<bit<32>>(5) local_util;     // Local util per port.
register<bit<48>>(5) last_packet_time;

header hulapp_t {
    bit<32>  targetID;    
    bit<32>  util; 
    bit<32>  version;   
    bit<32>  transTime; 
}

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;
typedef bit<16> digestType_t;

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    diffserv;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

struct ingress_metadata_t {

}

struct parser_metadata_t {
    bit<16>  remaining;
    bit<32>  dtag;      //tag of data packet 
    bit<32>  ptag;      //tag of probe packet 
}

struct traceroute_t{
    digestType_t type;
    ipv4_t ipv4;
}

struct metadata {
    /* empty */
    ingress_metadata_t   ingress_metadata;
    parser_metadata_t   parser_metadata;
    traceroute_t traceroute;
}

struct headers {
    ethernet_t                                      ethernet;
    ipv4_t                                                  ipv4;
    hulapp_t                                            hulapp;
}

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {

        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType){

            TYPE_IPV4: ipv4;
            default: accept;

        }
    }

    state ipv4 {
        packet.extract(hdr.ipv4);
        transition select (hdr.ipv4.protocol) {
            HULAPP_PROTOCOL      : parse_hulapp;
            _                    : accept;
        }
    }

    state parse_hulapp {
        packet.extract(hdr.hulapp);
        transition accept;
    }

}


/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}


/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    action drop() {
        mark_to_drop(standard_metadata);
    }


/*----------------------------------------------------------------------*/
/*If Hula++ probe, mcast it to the right set of next hops*/

    // Write to the standard_metadata's mcast field!
    action set_hulapp_mcast(bit<16> mcast_id) {
        hdr.hulapp.transTime=hdr.hulapp.transTime+1;
        standard_metadata.mcast_grp = mcast_id;
    }

    table hulapp_mcast {

        key = {
          standard_metadata.ingress_port : exact; 
        }

        actions = {
          set_hulapp_mcast; 
          drop; 
          NoAction; 
        }

        size = 1024;
        default_action = NoAction();
    }

    action send_digest() {
        meta.traceroute.type=0;
        meta.traceroute.ipv4=hdr.ipv4;
        digest(1, meta.traceroute);
        mark_to_drop(standard_metadata);
    }

    table exced_time_table {
        key = {
            hdr.ipv4.ttl: exact;
            // 你的条件字段
        }
        actions = {
            send_digest; // 定义一个总是调用 digest 的动作
            NoAction;
        }
        size = 4;
        default_action = NoAction();
    }



    action ipv4_forward(macAddr_t dstAddr, egressSpec_t port) {
        
        //decrease ttl by 1
        hdr.ipv4.ttl = hdr.ipv4.ttl -1;

        //set the src mac address as the previous dst
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;

        //set the destination mac address that we got from the match in the table
        hdr.ethernet.dstAddr = dstAddr;

        //set the output port that we also get from the table
        standard_metadata.egress_spec = port;

    }

    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    action update_util(bit<32> the_targetID,bit<32> the_util,bit<9> the_best_nhop){
        min_path_util.write(the_targetID,the_util);
        best_nhop.write(the_targetID,the_best_nhop);
    }

    apply {
        //only if IPV4 the rule is applied. Therefore other packets will not be forwarded.
        if (hdr.ipv4.isValid()){
            if (hdr.ipv4.protocol == HULAPP_PROTOCOL) {
                bit<32> the_version=hdr.hulapp.version;
                bit<32> the_targetID = hdr.hulapp.targetID;
                bit<32> the_util = hdr.hulapp.util;
                bit<32> local_version;
                version.read(local_version,the_targetID);
                if (the_version < local_version){
                    mark_to_drop(standard_metadata);
                }
                else{
                    bit<32> the_local_util;
                    local_util.read(the_local_util, (bit<32>)standard_metadata.ingress_port);
                    if (the_util < the_local_util){
                        the_util = the_local_util;
                        hdr.hulapp.util=the_util;
                    }
                    bit<32> the_min_path_util;
                    min_path_util.read(the_min_path_util,the_targetID);
                    if (the_version == local_version){
                        if (the_util<the_min_path_util){
                            update_util(the_targetID,the_util,standard_metadata.ingress_port);
                            hulapp_mcast.apply();
                        }
                        else{
                            mark_to_drop(standard_metadata);
                        }
                    }
                    else {
                        version.write(the_targetID,the_version);
                        update_util(the_targetID,the_util,standard_metadata.ingress_port);
                        hulapp_mcast.apply();
                    }
                }
            }
            else {
                ipv4_lpm.apply();

                exced_time_table.apply();
                // Update the path utilization if necessary
                if (standard_metadata.egress_spec != 1) {
                    bit<32> tmp_util = 0;
                    bit<48> tmp_time = 0;
                    bit<32> time_diff = 0;
                    local_util.read(tmp_util, (bit<32>) standard_metadata.egress_spec);
                    last_packet_time.read(tmp_time, (bit<32>) standard_metadata.egress_spec);
                    time_diff = (bit<32>)(standard_metadata.ingress_global_timestamp - tmp_time);
                    bit<32> temp = tmp_util*time_diff;
                    tmp_util = time_diff > UTIL_RESET_TIME_THRESHOLD ?
                                0 : standard_metadata.packet_length + tmp_util - (temp >> TAU_EXPONENT);
                    last_packet_time.write((bit<32>) standard_metadata.egress_spec,
                                            standard_metadata.ingress_global_timestamp);
                    local_util.write((bit<32>) standard_metadata.egress_spec, tmp_util);
                }
            }
        }


    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    apply { 
        
     }
}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
     apply {
	update_checksum(
	    hdr.ipv4.isValid(),
            { hdr.ipv4.version,
	      hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}


/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {

        //parsed headers have to be added again into the packet.
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.hulapp);

    }
}


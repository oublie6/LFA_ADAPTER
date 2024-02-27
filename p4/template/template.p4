// 定义数据包头部
header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header ipv4_t {
    bit<4> version;
    bit<4> ihl;
    bit<8> diffServ;
    bit<16> totalLen;
    bit<16> identification;
    bit<3> flags;
    bit<13> fragOffset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

header arp_t {
    bit<16> htype;
    bit<16> ptype;
    bit<8> hlen;
    bit<8> plen;
    bit<16> oper;
    bit<48> sha;
    bit<32> spa;
    bit<48> tha;
    bit<32> tpa;
}

// 定义数据包处理的流水线
parser MyParser(packet_in packet,
                out headers ethernet_t ethernet,
                out headers ipv4_t ipv4,
                out headers arp_t arp,
                inout metadata meta) {
    state start {
        packet.extract(ethernet);
        transition select(ethernet.etherType) {
            0x0800: parse_ipv4;
            0x0806: parse_arp;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(ipv4);
        transition accept;
    }

    state parse_arp {
        packet.extract(arp);
        transition accept;
    }
}

// 控制流水线的处理
control MyIngress(inout headers ethernet_t ethernet,
                  inout headers ipv4_t ipv4,
                  inout headers arp_t arp,
                  inout metadata meta) {
    action drop() {
        drop();
    }
    action send_icmp_error(bit<8> type, bit<8> code, packet_out packet, in headers ipv4_t ipv4) {
        // 构建 ICMP 响应包
        icmp_t icmp_resp;
        icmp_resp.type = type;
        icmp_resp.code = code;
        // 设置 ICMP 数据字段为原始 IP 数据包的头部和前 8 个字节
        icmp_resp.rest = ipv4;
        // 计算 ICMP 校验和
        // 这里需要填充 ICMP 校验和字段
        modify_field(icmp_resp.checksum, 0x0000); // Placeholder for checksum calculation
        // 修改 IPv4 头部 TTL 字段
        modify_field(ipv4.ttl, ipv4.ttl - 1);
        // 如果 TTL 减少为 0，则发送 ICMP Time Exceeded 错误
        if (ipv4.ttl == 0) {
            modify_field(icmp_resp.type, 11); // ICMP Time Exceeded
            modify_field(icmp_resp.code, 0);  // Code 0 for TTL exceeded during transit
        } else {
            // 如果路由表中没有找到下一跳地址，则发送 ICMP Destination Unreachable 错误
            modify_field(icmp_resp.type, 3); // ICMP Destination Unreachable
            modify_field(icmp_resp.code, 0);  // Code 0 for network unreachable
        }
        // 发送 ICMP 响应包
        packet.emit();
    }
    table ipv4_forward {
        key = {
            ipv4.dstAddr: lpm;
        }
        actions = {
            ipv4_forward,
            send_icmp_error;
        }
        size = 1024;
        default_action = send_icmp_error;
    }
    apply {
        if (ethernet.etherType == 0x0806) {
            apply(arp_table);
        } else if (ethernet.etherType == 0x0800) {
            apply(ipv4_forward);
        }
    }
}

# %%
import httpx
import dns.message
import re
import os

DOH_ENDPOINT = "https://cloudflare-dns.com/dns-query"
FRPC_INI_PATH = '~/frp_0.51.1_linux_amd64/frpc.ini'

def doh_query(domain):
    query = dns.message.make_query(domain, dns.rdatatype.A).to_wire()
    headers = {
        "accept": "application/dns-message",
        "content-type": "application/dns-message"
    }
    response = httpx.post(DOH_ENDPOINT, headers=headers, content=query)
    if response.status_code != 200:
        raise ValueError("DoH query failed!")
    dns_response = dns.message.from_wire(response.content)
    addresses = [answer.to_text().split()[-1] for answer in dns_response.answer if answer.rdtype == dns.rdatatype.A]
    return addresses[0] if addresses else None

def get_current_ip_from_frpc():
    with open(os.path.expanduser(FRPC_INI_PATH), 'r') as file:
        content = file.read()
        match = re.search(r'server_addr\s*=\s*(\d+\.\d+\.\d+\.\d+)', content)
        if match:
            return match.group(1)
    return None

def update_ip_in_frpc(new_ip):
    with open(os.path.expanduser(FRPC_INI_PATH), 'r') as file:
        content = file.read()
    updated_content = re.sub(r'server_addr\s*=\s*\d+\.\d+\.\d+\.\d+', f'server_addr = {new_ip}', content)
    with open(os.path.expanduser(FRPC_INI_PATH), 'w') as file:
        file.write(updated_content)

# def restart_frpc_service():
#     os.system('')

if __name__ == "__main__":
    try:
        new_ip = doh_query("whmc.ddns.net")
        current_ip = get_current_ip_from_frpc()

        if new_ip and new_ip != current_ip:
            print(f"IP address changed! Updating from {current_ip} to {new_ip}.")
            update_ip_in_frpc(new_ip)
            # restart_frpc_service()
            print("IP address updated!")
        else:
            print("IP address not changed!")

    except Exception as e:
        print(f"Error: {e}")


# %%

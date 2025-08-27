# utilities/consul_discovery.py
# =============================================================================
# 🎯 Purpose:
# A shared utility module for discovering Agent-to-Agent (A2A) servers
# It retrieves agent services from Consul registry and fetches
# each agent's metadata (AgentCard) to enable dynamic service discovery.
# =============================================================================

import json
import logging
import os
from typing import Any, Dict, List

import httpx
from httpx import QueryParams

from models.agent import AgentCard

# Create a named logger for this module
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ConsulDiscoveryClient:
    """
    🔍 Discovers A2A agents by querying the Consul registry and retrieving
    agent metadata as AgentCard objects.

    Attributes:
        consul_address (str): The address of the Consul server.
        consul_token (str): Authentication token for Consul (if required).
        id (str): Unique identifier for this discovery client instance to find the agents for this orchestrator.
        application_host (str): Host where the application is running.
        application_port (str): Port where the application is running.
    """

    def __init__(
        self,
        id: str,
        application_host: str,
        application_port: str,
        consul_address: str = None,
        consul_token: str = None,
    ) -> None:
        """
        Initialize the DiscoveryClient with Consul connection details.

        Args:
            id (str): Unique identifier for this discovery client instance to find the agents for this orchestrator.
            application_host (str): Host where the application is running.
            application_port (str): Port where the application is running.
            consul_address (str, optional): Consul server address. If None,
                defaults to environment variable or 'http://localhost:8500'.
            consul_token (str, optional): Consul authentication token.
        """
        self.consul_address = consul_address or os.environ.get(
            "CONSUL_HTTP_ADDR", "http://localhost:8500"
        )
        self.consul_token = consul_token or os.environ.get("CONSUL_HTTP_TOKEN", "")
        self.id = id
        self.application_host = application_host
        self.application_port = application_port
        self.networking = os.environ.get(
            "NETWORKING", "address"
        )  # "address" or "dns" or "transparent-proxy"
        logger.info(f"Consul networking is using: {self.networking}")

    async def watch_consul(self, callback=None, interval=60) -> None:
        """
        Continuously watch for changes in Consul services using blocking queries.
        When changes are detected, it calls the provided callback function with updated agent cards.

        Args:
            callback (Callable[[List[AgentCard]], None], optional): Function to call when services change.
                The function will receive a list of AgentCard objects.
            interval (int, optional): Fallback polling interval in seconds if blocking query fails.
                Defaults to 60 seconds.
        """
        if not callback:
            logger.warning(
                "Watching Consul services without a callback function has no effect."
            )
            return

        index = None  # Used for blocking queries

        while True:
            try:
                # Construct query parameters for blocking query
                params = {}
                headers = {}

                if self.consul_token:
                    headers["X-Consul-Token"] = self.consul_token

                if index:
                    params["index"] = index
                    params["wait"] = "30s"  # Wait up to 5 minutes for changes

                # Make blocking query to Consul
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.consul_address}/v1/catalog/services",
                        params=params,
                        headers=headers,
                        timeout=40.0,  # Slightly longer than wait time
                    )
                    response.raise_for_status()

                    # Get the updated index for the next blocking query
                    new_index = response.headers.get("X-Consul-Index")
                    if new_index:
                        index = new_index

                    # Check if services have changed
                    # services_data = response.json()
                    # print("Services data from Consul:", json.dumps(services_data, indent=2))

                    # Services changed, fetch updated agent cards
                    # logger.info("Service changes detected in Consul")
                    services = await self._get_services_from_consul()
                    agent_cards = await self.list_agent_cards(services=services)
                    mcp_servers = await self.list_mcp_servers(services=services)

                    # Call the callback with the updated agent cards
                    if callback:
                        try:
                            # Try calling with positional arguments first
                            await callback(
                                {"agents": agent_cards, "mcp_servers": mcp_servers}
                            )
                        except TypeError:
                            # If that fails, try with the named parameter
                            await callback(
                                subagents_card={
                                    "agents": agent_cards,
                                    "mcp_servers": mcp_servers,
                                }
                            )

            except Exception as e:
                logger.error(f"Error watching Consul services: {e}")
                index = None  # Reset index on error

                # Wait before retrying on error
                import asyncio

                await asyncio.sleep(interval)

    async def _get_services_from_consul(self) -> List[Any]:
        """
        Retrieve healthy services from Consul catalog.

        Returns:
            List[Dict[str, Any]]: List of healthy service details from Consul.
        """
        headers = {}
        if self.consul_token:
            headers["X-Consul-Token"] = self.consul_token

        services = []
        try:
            async with httpx.AsyncClient() as client:
                # First get all service names
                response = await client.get(
                    f"{self.consul_address}/v1/catalog/services",
                    headers=headers,
                    timeout=5.0,
                )
                response.raise_for_status()
                service_names = response.json()

                # Then get details for each service, but only include healthy ones
                for service_name in service_names:
                    try:
                        # Check if this service is allowed to talk to this service using Consul intentions
                        try:
                            intentions_check_url = (
                                f"{self.consul_address}/v1/connect/intentions/check"
                            )
                            intentions_payload = {
                                "source": self.id,
                                "destination": service_name,
                            }

                            params = QueryParams(intentions_payload)

                            intentions_headers = headers.copy()
                            async with httpx.AsyncClient() as intentions_client:
                                intentions_response = await intentions_client.get(
                                    intentions_check_url,
                                    params=params,
                                    headers=intentions_headers,
                                    timeout=5.0,
                                )
                                intentions_response.raise_for_status()
                                intentions_result = intentions_response.json()
                                if not intentions_result.get("Allowed", False):
                                    logger.debug(
                                        f"Intention denied: {self.id} -> {service_name}"
                                    )
                                    continue  # Skip this instance if not allowed
                        except Exception as e:
                            logger.warning(
                                f"Failed to check intentions for {self.id} -> {service_name}: {e}"
                            )
                            continue

                        # Get service health status - this includes health check info
                        health_response = await client.get(
                            f"{self.consul_address}/v1/health/service/{service_name}",
                            headers=headers,
                            timeout=5.0,
                        )
                        health_response.raise_for_status()
                        health_data = health_response.json()

                        # Filter to only include healthy instances
                        healthy_instances = []
                        for instance in health_data:
                            # Check if all health checks are passing
                            is_healthy = all(
                                check.get("Status") == "passing"
                                for check in instance.get("Checks", [])
                            )
                            if is_healthy:
                                service_data = instance.get("Service", {})

                                # Transform to the expected format
                                transformed_service = {
                                    "ServiceID": service_data.get("ID", "N/A"),
                                    "ServiceName": service_data.get("Service", "N/A"),
                                    "ServiceAddress": service_data.get(
                                        "Address", "127.0.0.1"
                                    ),
                                    "ServicePort": service_data.get("Port", "80"),
                                    "ServiceMeta": service_data.get("Meta", {}),
                                    "ServiceTags": service_data.get("Tags", []),
                                }
                                healthy_instances.append(transformed_service)
                                # if one healthy instance is found, we can stop checking further instances
                                break

                        if not healthy_instances:
                            logger.warning(
                                f"No healthy instances found for service {service_name}."
                            )
                            continue

                        services.extend(healthy_instances)
                    except Exception as e:
                        logger.warning(
                            f"Failed to get details for service {service_name}: {e}"
                        )

            # print the services for debugging
            # print("Healthy services retrieved from Consul:", json.dumps(services, indent=2))
            return services
        except Exception as e:
            logger.error(f"Failed to retrieve services from Consul: {e}")
            return []

    # async def get_services_by_meta(self, meta_key: str, meta_value: str) -> List[Dict[str, Any]]:
    #     """
    #     Retrieve services from Consul filtered by a specific metadata key-value pair.
    #
    #     Args:
    #         meta_key (str): The metadata key to filter on
    #         meta_value (str): The value to match
    #
    #     Returns:
    #         List[Dict[str, Any]]: List of services matching the metadata criteria
    #     """
    #     all_services = await self._get_services_from_consul()
    #
    #     # Filter services by metadata
    #     matching_services = []
    #     for service in all_services:
    #         if "ServiceMeta" in service and meta_key in service["ServiceMeta"]:
    #             if service["ServiceMeta"][meta_key] == meta_value:
    #                 matching_services.append(service)
    #
    #     return matching_services

    async def list_agent_cards(self, services: List[Any]) -> List[AgentCard]:
        """
        Asynchronously fetch agent services from Consul and convert
        them to AgentCard objects.

        Returns:
            List[AgentCard]: Successfully retrieved agent cards.
        """
        cards: List[AgentCard] = []

        async with httpx.AsyncClient() as client:
            for service in services:
                # if USE_DNS is set to True, use DNS-based service discovery
                if os.environ.get("USE_DNS") == "TRUE":
                    # Construct URL and fetch agent metadata
                    url = f"http://{service}:{self.application_port}/.well-known/agent.json"
                    print(f"Url for agent discovery: {url}")
                else:
                    # Skip services without proper metadata
                    if "ServiceMeta" not in service:
                        continue

                    if "agent-type" not in service["ServiceMeta"]:
                        logger.debug(
                            f"Service {service.get('ServiceName', 'unknown')} does not have 'agent-type' metadata. Skipping."
                        )
                        continue

                    if service["ServiceMeta"]["agent-type"] != "ai-agent":
                        logger.debug(
                            f"Service {service.get('ServiceName', 'unknown')} is not an AI agent. Skipping."
                        )
                        continue

                    # Try to get service address and port
                    address = service.get("ServiceAddress")
                    port = service.get("ServicePort")

                    if not address or not port:
                        continue

                    # Construct URL and fetch agent metadata
                    url = ""
                    if self.networking == "dns":
                        url = f"http://{service.get('ServiceName')}.service.consul:{port}/.well-known/agent.json"
                    elif self.networking == "address":
                        url = f"http://{address}:{port}/.well-known/agent.json"
                    elif self.networking == "transparent-proxy":
                        url = f"http://{service.get('ServiceName')}.virtual.consul/.well-known/agent.json"
                    else:
                        url = f"http://{address}:{port}/.well-known/agent.json"

                # get the agent card from the service URL
                try:
                    response = await client.get(url, timeout=5.0)
                    response.raise_for_status()
                    card = AgentCard.model_validate(response.json())
                    cards.append(card)
                except Exception as e:
                    logger.debug(f"Failed to discover agent at {url}: {e}")

        return cards

    async def list_mcp_servers(self, services: List[Any]) -> List[Any]:
        """
        Asynchronously fetch agent services from Consul and convert
        them to MCP metadata objects.

        Returns:
            List[Any]: Successfully retrieved mcp servers.
        """
        cards: List[Any] = []

        async with httpx.AsyncClient() as client:
            for service in services:
                # if USE_DNS is set to True, use DNS-based service discovery
                if os.environ.get("USE_DNS") == "TRUE":
                    # Construct URL and fetch agent metadata
                    url = f"http://{service}:{self.application_port}/.well-known/agent.json"
                    print(f"Url for agent discovery: {url}")
                else:
                    # Skip services without proper metadata
                    if "ServiceMeta" not in service:
                        continue

                    if "agent-type" not in service["ServiceMeta"]:
                        logger.debug(
                            f"Service {service.get('ServiceName', 'unknown')} does not have 'agent-type' metadata. Skipping."
                        )
                        continue

                    if service["ServiceMeta"]["agent-type"] != "ai-mcp":
                        logger.debug(
                            f"Service {service.get('ServiceName', 'unknown')} is not an AI mcp. Skipping."
                        )
                        continue

                    # Try to get service address and port
                    address = service.get("ServiceAddress")
                    port = service.get("ServicePort")

                    if not address or not port:
                        continue

                    # Construct URL and fetch agent metadata
                    url = ""
                    if self.networking == "dns":
                        url = f"http://{service.get('ServiceName')}.service.consul:{port}/sse"
                    elif self.networking == "address":
                        url = f"http://{address}:{port}/sse"
                    elif self.networking == "transparent-proxy":
                        url = f"http://{service.get('ServiceName')}.virtual.consul/sse"
                    else:
                        url = f"http://{address}:{port}/sse"

                    cards.append(
                        {
                            "name": service.get("ServiceName"),
                            "address": service.get("ServiceAddress"),
                            "port": service.get("ServicePort"),
                            "meta": service.get("ServiceMeta", {}),
                            "tags": service.get("ServiceTags", []),
                            "url": url,
                        }
                    )

        return cards

    def register_agent(self, agent: AgentCard) -> None:
        """
        Register an agent with Consul.

        Args:
            agent (AgentCard): The agent to register.
        """
        # Use the instance variables instead of parameters
        consul_address = self.consul_address
        consul_token = self.consul_token
        parent_ai_agent_id = os.environ.get(
            "PARENT_AI_AGENT_ID", default="orchestrator_agent"
        )

        headers = {}
        if consul_token:
            headers["X-Consul-Token"] = consul_token

        service_data = {
            "Name": agent.id,
            "ID": agent.id + "-" + self.application_host,
            "Address": self.application_host,
            "Port": self.application_port,
            "Meta": {
                "description": agent.description,
                "version": agent.version,
                "parent_ai_agent": parent_ai_agent_id,
                "ai-agent": "true",
            },
            "EnableTagOverride": False,
            "Checks": [
                {
                    "Name": "HTTP Health Check",
                    "HTTP": f"http://{self.application_host}:{self.application_port}/health",
                    "Interval": "10s",
                    "Timeout": "1s",
                }
            ],
        }

        # print the service data for debugging
        print("Service data to register:", json.dumps(service_data, indent=2))

        try:
            response = httpx.put(
                f"{consul_address}/v1/agent/service/register",
                json=service_data,
                headers=headers,
                timeout=5.0,
            )
            response.raise_for_status()
            logger.info(f"Agent {agent.name} registered successfully with Consul.")
        except httpx.RequestError as e:
            logger.error(f"Failed to register agent {agent.name} with Consul: {e}")

    async def get_kv_variable(self, variable: str) -> str | None:
        """
        Fetch a variable value from Consul KV store using the path pattern:
        consul-adk/{self.id}/variables/{variable}

        Args:
            variable (str): The variable name to fetch from the KV store.

        Returns:
            str | None: The value of the variable if found, None if not found or on error.

        Raises:
            httpx.HTTPStatusError: If the HTTP request fails with a status error.
            Exception: For other connection or parsing errors.
        """
        kv_path = f"consul-adk/{self.id}/variables/{variable}"
        logger.info(f"Fetching {variable} from KV store at path: {kv_path}")

        try:
            headers = {}
            if self.consul_token:
                headers["X-Consul-Token"] = self.consul_token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.consul_address}/v1/kv/{kv_path}",
                    headers=headers,
                    timeout=30.0,
                )

                # If key doesn't exist, Consul returns 404
                if response.status_code == 404:
                    logger.warning(f"KV variable '{variable}' not found at path: {kv_path}")
                    return None

                response.raise_for_status()

                # Parse the response
                kv_data = response.json()

                if not kv_data or len(kv_data) == 0:
                    logger.warning(f"Empty response for KV variable '{variable}' at path: {kv_path}")
                    return None

                # Consul KV API returns a list of objects, get the first one
                kv_entry = kv_data[0]

                # The value is base64 encoded, decode it
                import base64

                encoded_value = kv_entry.get("Value")
                if encoded_value:
                    decoded_value = base64.b64decode(encoded_value).decode("utf-8")
                    logger.info(f"Successfully retrieved KV variable '{variable}' from path: {kv_path}")
                    return decoded_value
                else:
                    logger.warning(f"KV variable '{variable}' has no value at path: {kv_path}")
                    return None

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching KV variable '{variable}' from path '{kv_path}': {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching KV variable '{variable}' from path '{kv_path}': {e}")
            raise


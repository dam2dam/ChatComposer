package com.a504.chatcomposer;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.servers.Server;

@OpenAPIDefinition(servers = {
	@Server(url = "https://k8a504.p.ssafy.io/api/", description = "ChatComposer Server URL"),
	@Server(url = "http://localhost:8080/api/", description = "local URL")
})
@SpringBootApplication
public class ChatComposerApplication {

	public static void main(String[] args) {
		SpringApplication.run(ChatComposerApplication.class, args);
	}
}

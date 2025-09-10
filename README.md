# Ansible-H.IAAC


Este repositório contém o conjunto de playbooks e configurações para provisionar e gerenciar um testbed para experimentos de Aprendizagem Federada. O ambiente é composto por um cluster **heterogêneo** de mini-computadores (Raspberry Pi e NVIDIA Jetson), orquestrado via Ansible e operando em uma sub-rede isolada.


O objetivo principal é criar uma plataforma que simula **cenários de borda (_edge computing_) do mundo real**, permitindo avaliar o desempenho de diferentes algoritmos em dispositivos com diversas restrições de hardware (capacidade de processamento, consumo de energia, dissipação térmica, etc.). A automação com Ansible garante que os experimentos sejam **reproduzíveis, escaláveis e fáceis de gerenciar**, acelerando o ciclo de pesquisa e eliminando configurações manuais repetitivas.
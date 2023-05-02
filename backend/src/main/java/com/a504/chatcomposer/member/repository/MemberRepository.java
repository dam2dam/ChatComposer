package com.a504.chatcomposer.member.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.a504.chatcomposer.member.entity.Member;

@Repository
public interface MemberRepository extends JpaRepository<Member, Long> {
}

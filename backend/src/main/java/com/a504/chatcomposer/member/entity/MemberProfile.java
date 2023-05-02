package com.a504.chatcomposer.member.entity;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.MapsId;
import javax.persistence.OneToOne;
import javax.persistence.Table;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Builder
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@Entity
@Table(name = "member_profile")
public class MemberProfile {

	@Id
	@Column(name = "member_id", nullable = false)
	private Long id;

	@MapsId    // Profile의 memberId와 매핑 (1:1 식별관계)
	@OneToOne
	@JoinColumn(name = "member_id", nullable = false)
	private Member member;
}

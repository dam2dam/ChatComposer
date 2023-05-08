package com.a504.chatcomposer.music.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.EnumType;
import javax.persistence.Enumerated;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.OneToMany;
import javax.persistence.Table;
import javax.validation.constraints.NotNull;

import org.hibernate.annotations.ColumnDefault;
import org.hibernate.annotations.CreationTimestamp;

import com.a504.chatcomposer.member.entity.FavoriteMusic;
import com.a504.chatcomposer.member.entity.Member;
import com.a504.chatcomposer.music.dto.enums.Beat;
import com.a504.chatcomposer.music.dto.enums.Genre;

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
@Table(name = "music")
public class Music {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "music_id", nullable = false)
	private Long id;

	@NotNull
	@ManyToOne
	@JoinColumn(name = "member_id")
	private Member member;

	@Column(name = "title", length = 20)
	private String title;

	@Column(name = "description", length = 200)
	private String description;

	@Column(name = "music_source", length = 200)
	private String musicSource;

	@Column(name = "mixed_music_source", length = 200)
	private String mixedMusicSource;

	@Column(name = "cover_source", length = 200)
	private String coverSource;

	@NotNull
	@Column(name = "favorite_count")
	@ColumnDefault("0")
	private int favoriteCount;

	@NotNull
	@Enumerated(EnumType.STRING)
	@Column(name = "genre")
	private Genre genre;

	@NotNull
	@Enumerated(EnumType.STRING)
	@Column(name = "beat")
	private Beat beat;

	@Column(name = "mixed_music_request", length = 500)
	private String mixedMusicRequest;

	@Column(name = "cover_request", length = 500)
	private String coverRequest;

	@Column(name = "created_at", nullable = false)
	@CreationTimestamp
	@ColumnDefault("CURRENT_TIMESTAMP")
	private LocalDateTime createdAt;

	@OneToMany(mappedBy = "music")
	private List<Track> tracks = new ArrayList<>();

	@OneToMany(mappedBy = "music")
	private List<FavoriteMusic> favoriteMusics = new ArrayList<>();

	@OneToMany(mappedBy = "music")
	private List<MusicTag> musicTags = new ArrayList<>();
}
